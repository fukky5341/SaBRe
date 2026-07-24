## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 607.026092835328


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-317.7158813, 374.1755066, -317.7158813, 374.1755066, -691.8913574, 691.8913574)
1: (-264.8054199, 302.6636658, -264.8054199, 302.6636658, -567.4689331, 567.4689331)
2: (-213.3653564, 297.9397583, -213.3653564, 297.9397583, -511.3051147, 511.3051147)
3: (-300.8116760, 377.7047119, -300.8116760, 377.7047119, -678.5163574, 678.5163574)
4: (-274.6094055, 403.1719055, -274.6094055, 403.1719055, -677.7813110, 677.7813110)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.92 + 2.40 = 4.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -607.0382336, upper bound: 607.0382336

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295249, upper bound: 607.0362170
time: 0.83 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -607.0295249, upper bound: 607.0362170
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -295.9215698, 347.1386719, -317.7158813, 374.1755066, -670.0970459, 664.8545532
1: -246.2148590, 280.6079407, -264.8054199, 302.6636658, -548.8784180, 545.4133301
2: -198.3210297, 275.8583069, -213.3653564, 297.9397583, -496.2607727, 489.2236633
3: -278.7344971, 350.5451355, -300.8116760, 377.7047119, -656.4392090, 651.3567505
4: -255.0973816, 373.1489563, -274.6094055, 403.1719055, -658.2692871, 647.7583618

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
time: 1.02 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
time: 0.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -453.9238586, 526.4202881, -314.6738281, 369.9300537, -822.1127319, 841.0941162
1: -384.9192200, 425.2345581, -262.1333618, 299.2684326, -680.4078369, 687.3677368
2: -304.9117126, 418.6582642, -211.2179260, 294.4729919, -598.3780518, 629.5482788
3: -425.5919189, 534.1150513, -297.4461060, 373.6198120, -799.2116699, 830.3502808
4: -389.1732483, 567.7148438, -271.7906799, 398.4138184, -786.7455444, 839.0061646

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
time: 1.05 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
time: 0.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.89 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.89
Output dim: 0, lower bound: -607.0295076, upper bound: 607.0295076

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -295.9215698, 347.1386719, -295.9215698, 347.1386719, -643.0602417, 643.0602417
1: -246.2148590, 280.6079407, -246.2148590, 280.6079407, -526.8228149, 526.8228149
2: -198.3210297, 275.8583069, -198.3210297, 275.8583069, -474.1793213, 474.1793213
3: -278.7344971, 350.5451355, -278.7344971, 350.5451355, -629.2796021, 629.2796021
4: -255.0973816, 373.1489563, -255.0973816, 373.1489563, -628.2462769, 628.2462769

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295123, upper bound: 607.0362090
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302295
time: 1.06 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -295.9215698, 347.1386719, -449.7806396, 521.0416870, -816.9632568, 795.1713257
1: -246.2148590, 280.6079407, -381.4551086, 420.9349060, -667.1497803, 658.2518921
2: -198.3210297, 275.8583069, -302.0771484, 414.3644714, -612.3428345, 576.9490967
3: -278.7344971, 350.5451355, -421.4091797, 528.8980713, -806.4389038, 771.9543457
4: -255.0973816, 373.1489563, -385.4582214, 561.9293213, -816.5632324, 757.8134766

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290731, upper bound: 607.0301627
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302355
time: 0.99 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -449.7709351, 521.0288696, -294.8175049, 345.8101501, -793.8328857, 815.8463135
1: -381.4450684, 420.9249268, -245.2889557, 279.5309143, -657.1652832, 666.2138672
2: -302.0699158, 414.3543701, -197.5696106, 274.7982788, -575.8818970, 611.5816650
3: -421.3983154, 528.8849487, -277.6728210, 349.2039185, -770.6022339, 805.3645020
4: -385.4490662, 561.9156494, -254.1246643, 371.7185059, -756.3740234, 815.5748291

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0294640, upper bound: 607.0290717
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295060, upper bound: 607.0295022
time: 0.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -472.1424255, 549.9840698, -472.1424255, 549.9840698, -1018.9993896, 1018.9993896
1: -400.1933899, 444.1636658, -400.1933899, 444.1636658, -839.4248047, 839.4248047
2: -317.4039001, 437.8050232, -317.4039001, 437.8050232, -753.1975708, 753.1975708
3: -444.4273376, 556.9395142, -444.4273376, 556.9395142, -999.5861206, 999.5861206
4: -405.4700928, 593.5077515, -405.4700928, 593.5077515, -996.5107422, 996.5106812

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290754, upper bound: 607.0294640
time: 1.02 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295060, upper bound: 607.0295022
time: 1.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.39 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0295123, upper bound: 607.0362090
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302295
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0290731, upper bound: 607.0301627
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302355
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0294640, upper bound: 607.0290717
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0295060, upper bound: 607.0295022
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0290754, upper bound: 607.0294640
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 0, lower bound: -607.0295060, upper bound: 607.0295022

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -286.4641724, 335.2057190, -295.9215698, 347.1386719, -633.6028442, 631.1273193
1: -238.0624084, 270.9075623, -246.2148590, 280.6079407, -518.6703491, 517.1224365
2: -191.7396545, 266.1152344, -198.3210297, 275.8583069, -467.5979614, 464.4361877
3: -269.0568237, 338.5891724, -278.7344971, 350.5451355, -619.6019287, 617.3236694
4: -246.5367889, 359.9270935, -255.0973816, 373.1489563, -619.6857300, 615.0244751

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -458.7105713, 538.9670410, -293.6995239, 344.3240662, -802.3648682, 832.1066895
1: -387.2208862, 433.9443359, -244.3365631, 278.3478088, -662.9113159, 678.2808228
2: -307.1966858, 428.2314453, -196.8003235, 273.6096802, -580.4002075, 624.5629272
3: -432.6846924, 544.0994873, -276.5317383, 347.7354736, -780.4201660, 819.7779541
4: -393.2234497, 581.0760498, -253.1131134, 370.1144714, -763.3239746, 833.4262085

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -295.9215698, 347.1386719, -447.8724365, 518.6846924, -814.4945679, 793.0617065
1: -246.2148590, 280.6079407, -379.6879883, 418.7928467, -664.9710693, 656.1349487
2: -198.3210297, 275.8583069, -300.5069580, 412.0950928, -609.9832764, 575.2868652
3: -278.7344971, 350.5451355, -419.2807312, 526.1636353, -803.4526367, 769.8258667
4: -255.0973816, 373.1489563, -383.5152893, 558.6697998, -813.2163086, 755.7179565

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290731, upper bound: 607.0301627
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290731, upper bound: 607.0301627
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -293.6995239, 344.3240662, -525.9685059, 611.1761475, -902.8593140, 865.7536621
1: -244.3365631, 278.3478088, -451.5503540, 492.3901062, -735.5952759, 721.0803223
2: -196.8003235, 273.6096802, -352.9429626, 485.9607239, -681.2053833, 624.5714111
3: -276.5317383, 347.7354736, -494.1235962, 621.0208130, -894.2066040, 841.4063721
4: -253.1131134, 370.1144714, -449.0488281, 660.7403564, -911.4050293, 817.4938354

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302355
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302355
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -447.8645325, 518.6742554, -294.8175049, 345.8101501, -791.7250977, 813.3791504
1: -379.6799316, 418.7846069, -245.2889557, 279.5309143, -655.0502319, 664.0360107
2: -300.5011597, 412.0866699, -197.5696106, 274.7982788, -574.2209473, 609.2236938
3: -419.2717285, 526.1529541, -277.6728210, 349.2039185, -768.4756470, 802.3806152
4: -383.5079346, 558.6583252, -254.1246643, 371.7185059, -754.2802124, 812.2301025

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301627, upper bound: 607.0290731
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301627, upper bound: 607.0290731
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -525.9406738, 611.1382446, -292.6150513, 343.0423889, -864.4439697, 901.7366333
1: -451.5281067, 492.3591919, -243.4366455, 277.3081055, -720.0173950, 734.6644897
2: -352.9237061, 485.9321289, -196.0683289, 272.5887756, -623.5312500, 680.4451294
3: -494.0950623, 620.9855957, -275.5069580, 346.4385376, -840.0800171, 893.1459351
4: -449.0229187, 660.7037354, -252.1687775, 368.7367249, -816.0908203, 910.4208984

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302355, upper bound: 607.0295037
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302355, upper bound: 607.0295037
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -472.1424255, 549.9840698, -464.9393005, 541.0213623, -1009.9130859, 1011.6419678
1: -400.1933899, 444.1636658, -394.1322937, 436.8084106, -831.9586792, 833.1021729
2: -317.4039001, 437.8050232, -312.2680054, 430.4052429, -745.7359619, 748.0191650
3: -444.4273376, 556.9395142, -437.1111755, 547.9950562, -990.4699097, 992.2443848
4: -405.4700928, 593.5077515, -398.8383484, 583.4384766, -986.3732910, 989.8073120

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290352, upper bound: 607.0290352
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290352, upper bound: 607.0294640
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -457.4489136, 532.3990479, -568.2802124, 666.0757446, -1118.1785889, 1094.5163574
1: -388.1774902, 429.9313965, -487.9352722, 536.3952026, -918.1738281, 907.9010010
2: -307.6258850, 423.7109985, -382.3090515, 530.0568848, -834.4248657, 802.9308472
3: -430.2981873, 539.3264160, -537.7401123, 674.2736206, -1100.5860596, 1074.5438232
4: -392.8003845, 574.4367065, -486.9641418, 720.2551880, -1108.6390381, 1057.9967041

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0294640, upper bound: 607.0290717
time: 1.02 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0294640, upper bound: 607.0295022
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.15 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0302313, upper bound: 607.0302313
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0290731, upper bound: 607.0301627
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0290731, upper bound: 607.0301627
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302355
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0295037, upper bound: 607.0302355
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0301627, upper bound: 607.0290731
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0301627, upper bound: 607.0290731
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0302355, upper bound: 607.0295037
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0302355, upper bound: 607.0295037
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0290352, upper bound: 607.0290352
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0290352, upper bound: 607.0294640
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0294640, upper bound: 607.0290717
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.15
Output dim: 0, lower bound: -607.0294640, upper bound: 607.0295022

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -286.4641724, 335.2057190, -286.4641724, 335.2057190, -621.6699219, 621.6699219
1: -238.0624084, 270.9075623, -238.0624084, 270.9075623, -508.9699097, 508.9699097
2: -191.7396545, 266.1152344, -191.7396545, 266.1152344, -457.8548584, 457.8548889
3: -269.0568237, 338.5891724, -269.0568237, 338.5891724, -607.6459961, 607.6459961
4: -246.5367889, 359.9270935, -246.5367889, 359.9270935, -606.4638672, 606.4638672

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302108, upper bound: 607.0362741
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302244, upper bound: 607.0331396
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -286.4641724, 335.2057190, -458.0378113, 538.0620117, -823.9942017, 792.5576172
1: -238.0624084, 270.9075623, -386.6437683, 433.2222290, -671.2846680, 654.8750000
2: -191.7396545, 266.1152344, -306.7275391, 427.5044556, -618.7547607, 572.4418945
3: -269.0568237, 338.5891724, -432.0056152, 543.2086182, -811.4049072, 770.5947876
4: -246.5367889, 359.9270935, -392.6040344, 580.1060181, -825.8909302, 752.5311279

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302108, upper bound: 607.0362741
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0302244, upper bound: 607.0331408
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -458.0378113, 538.0620117, -286.4641724, 335.2057190, -792.5575562, 823.9942627
1: -386.6437683, 433.2222290, -238.0624084, 270.9075623, -654.8750000, 671.2846680
2: -306.7275391, 427.5044556, -191.7396545, 266.1152344, -572.4418945, 618.7547607
3: -432.0056152, 543.2086182, -269.0568237, 338.5891724, -770.5947876, 811.4049072
4: -392.6040344, 580.1060181, -246.5367889, 359.9270935, -752.5311279, 825.8909302

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0292979, upper bound: 607.0274350
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0284962, upper bound: 607.0300584
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301002, upper bound: 607.0301002
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -461.3457642, 542.5102539, -461.3457642, 542.5102539, -1001.3272705, 1001.3272705
1: -389.4883728, 436.7746277, -389.4883728, 436.7746277, -822.6694336, 822.6694336
2: -309.0398560, 431.0817566, -309.0398560, 431.0817566, -738.6675415, 738.6675415
3: -435.3831787, 547.5911255, -435.3831787, 547.5911255, -981.6538696, 981.6538696
4: -395.6549377, 584.8927002, -395.6549377, 584.8927002, -978.7233887, 978.7233887

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300584, upper bound: 607.0284962
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301002, upper bound: 607.0301002
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -286.4641724, 335.2057190, -445.5693665, 515.6999512, -802.0791016, 778.8314209
1: -238.0624084, 270.9075623, -377.7279053, 416.3992310, -654.4616699, 644.4613037
2: -191.7396545, 266.1152344, -298.9267578, 409.6824036, -600.9685669, 563.9758301
3: -269.0568237, 338.5891724, -416.9118958, 523.2530518, -790.8746948, 755.5010986
4: -246.5367889, 359.9270935, -381.4467163, 555.4244995, -801.4172363, 740.4606323

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0235700, upper bound: 607.0289730
time: 1.01 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275058, upper bound: 607.0298367
time: 0.91 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0290573, upper bound: 607.0301221
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -461.3457642, 542.5102539, -462.0101318, 537.2123413, -996.4487915, 1000.6966553
1: -389.4883728, 436.7746277, -391.6236267, 433.7262573, -819.3125610, 823.3713379
2: -309.0398560, 431.0817566, -310.2124939, 427.2598877, -734.8806152, 739.1398315
3: -435.3831787, 547.5911255, -434.0064087, 544.2599487, -977.7440186, 980.1908569
4: -395.6549377, 584.8927002, -396.1819153, 579.1587524, -973.2200317, 978.2388306

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275607, upper bound: 607.0299895
time: 0.83 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0286490, upper bound: 607.0300330
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -286.4641724, 335.2057190, -521.5147705, 605.5243530, -890.0499268, 852.2263794
1: -238.0624084, 270.9075623, -447.7488708, 487.9078369, -724.9121704, 709.8411865
2: -191.7396545, 266.1152344, -349.8837891, 481.5068665, -671.6890259, 614.0513916
3: -269.0568237, 338.5891724, -489.6702271, 615.5280762, -881.2606201, 827.8151245
4: -246.5367889, 359.9270935, -445.1008911, 654.7503662, -898.8581543, 803.4160156

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0235700, upper bound: 607.0301781
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265836, upper bound: 607.0291951
time: 0.95 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265568, upper bound: 607.0273079
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -461.3457642, 542.5102539, -557.1105347, 651.8006592, -1108.8287354, 1092.8681641
1: -389.4883728, 436.7746277, -478.4169312, 524.8934326, -909.1425171, 905.3235474
2: -309.0398560, 431.0817566, -374.5802612, 518.4171143, -824.8037720, 802.4522095
3: -435.3831787, 547.5911255, -526.1879883, 660.3656006, -1091.7741699, 1071.5747070
4: -395.6549377, 584.8927002, -476.9458618, 704.4866943, -1096.5987549, 1058.1304932

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265836, upper bound: 607.0291342
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265568, upper bound: 607.0271905
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -445.5608215, 515.6887207, -285.3448181, 333.8539124, -777.4713135, 800.9483643
1: -377.7192688, 416.3905945, -237.1194763, 269.8140259, -643.3585815, 653.5096436
2: -298.9205627, 409.6734619, -190.9758759, 265.0350037, -562.8894653, 600.1954956
3: -416.9023132, 523.2416992, -267.9738159, 337.2276611, -754.1300049, 789.7809448
4: -381.4388428, 555.4122314, -245.5482330, 358.4687500, -738.9948730, 800.4146729

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0289730, upper bound: 607.0235700
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0298367, upper bound: 607.0275058
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301221, upper bound: 607.0290573
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -462.0101318, 537.2123413, -449.8073120, 528.4021606, -986.5656738, 984.8800659
1: -391.6236267, 433.7262573, -379.9004211, 425.3961182, -811.9819336, 809.6813965
2: -310.2124939, 427.2598877, -301.2584534, 419.8677368, -727.9142456, 727.0664062
3: -434.0064087, 544.2599487, -424.1621704, 533.4725342, -966.0373535, 966.4945068
4: -396.1819153, 579.1587524, -385.5106812, 569.7819214, -963.1076050, 963.0249634

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0299895, upper bound: 607.0275607
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300330, upper bound: 607.0286490
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -521.4987793, 605.5014038, -285.3448181, 333.8539124, -850.8583984, 888.9081421
1: -447.7372437, 487.8890076, -237.1194763, 269.8140259, -708.7340698, 723.9507446
2: -349.8728638, 481.4902954, -190.9758759, 265.0350037, -612.9603271, 670.9082642
3: -489.6534119, 615.5075073, -267.9738159, 337.2276611, -826.4363403, 880.1566162
4: -445.0853577, 654.7299805, -245.5482330, 358.4687500, -801.9434814, 897.8465576

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0301781, upper bound: 607.0273176
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0291951, upper bound: 607.0268344
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273078, upper bound: 607.0268076
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -557.6938477, 652.5394897, -449.8073120, 528.4021606, -1079.3205566, 1097.9974365
1: -478.9078369, 525.4862671, -379.9004211, 425.3961182, -894.4310913, 900.1029663
2: -374.9784241, 519.0155640, -301.2584534, 419.8677368, -791.6262207, 817.5878906
3: -526.7798462, 661.0822754, -424.1621704, 533.4725342, -1058.0152588, 1081.2424316
4: -477.4617920, 705.2965088, -385.5106812, 569.7819214, -1043.5147705, 1087.2145996

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0291950, upper bound: 607.0268344
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273079, upper bound: 607.0268076
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -464.9393005, 541.0213623, -464.9393005, 541.0213623, -1002.5556641, 1002.5556641
1: -394.1322937, 436.8084106, -394.1322937, 436.8084106, -825.6359863, 825.6359863
2: -312.2680054, 430.4052429, -312.2680054, 430.4052429, -740.5575562, 740.5575562
3: -437.1111755, 547.9950562, -437.1111755, 547.9950562, -983.1281738, 983.1281738
4: -398.8383484, 583.4384766, -398.8383484, 583.4384766, -979.6699219, 979.6699219

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0234507, upper bound: 607.0263143
time: 0.90 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0233075
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -568.2888184, 666.0836182, -464.9393005, 541.0213623, -1102.9569092, 1125.3854980
1: -487.9415894, 536.4021606, -394.1322937, 436.8084106, -914.6472778, 923.8712769
2: -382.3149719, 530.0637817, -312.2680054, 430.4052429, -809.5533447, 838.9750977
3: -537.7482910, 674.2809448, -437.1111755, 547.9950562, -1082.9957275, 1107.3356934
4: -486.9710083, 720.2650146, -398.8383484, 583.4384766, -1066.9187012, 1114.5599365

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266371, upper bound: 607.0272115
time: 1.07 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0270165
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -464.9393005, 541.0213623, -567.7958984, 665.4495850, -1124.7523193, 1102.4635010
1: -394.1322937, 436.8084106, -487.5209045, 535.8898315, -923.3596802, 914.2218628
2: -312.2680054, 430.4052429, -381.9725952, 529.5443726, -838.4556885, 809.2099609
3: -437.1111755, 547.9950562, -537.2343140, 673.6604004, -1106.7144775, 1082.4802246
4: -398.8383484, 583.4384766, -486.5245667, 719.5631104, -1113.8568115, 1066.4738770

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0285892, upper bound: 607.0275603
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0285961, upper bound: 607.0286472
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -570.2347412, 668.6093140, -570.2347412, 668.6093140, -1230.2662354, 1230.2662354
1: -489.6099243, 538.4390869, -489.6099243, 538.4390869, -1016.6116333, 1016.6116333
2: -383.6712646, 532.1336670, -383.6712646, 532.1336670, -911.4058838, 911.4058838
3: -539.7968140, 676.7504883, -539.7968140, 676.7504883, -1211.7373047, 1211.7373047
4: -488.7454834, 723.0624390, -488.7454834, 723.0624390, -1206.3813477, 1206.3813477

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0234507, upper bound: 607.0282727
time: 0.90 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0272571
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.86 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0302108, upper bound: 607.0362741
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0302244, upper bound: 607.0331396
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0302108, upper bound: 607.0362741
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0302244, upper bound: 607.0331408
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0284962, upper bound: 607.0300584
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0301002, upper bound: 607.0301002
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0300584, upper bound: 607.0284962
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0301002, upper bound: 607.0301002
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0275058, upper bound: 607.0298367
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0290573, upper bound: 607.0301221
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0275607, upper bound: 607.0299895
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0286490, upper bound: 607.0300330
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0265836, upper bound: 607.0291951
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0265568, upper bound: 607.0273079
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0265836, upper bound: 607.0291342
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0265568, upper bound: 607.0271905
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0298367, upper bound: 607.0275058
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0301221, upper bound: 607.0290573
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0299895, upper bound: 607.0275607
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0300330, upper bound: 607.0286490
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0291951, upper bound: 607.0268344
NS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0273078, upper bound: 607.0268076
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0291950, upper bound: 607.0268344
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0273079, upper bound: 607.0268076
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0234507, upper bound: 607.0263143
NS_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0233075
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0266371, upper bound: 607.0272115
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0270165
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0285892, upper bound: 607.0275603
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0285961, upper bound: 607.0286472
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0234507, upper bound: 607.0282727
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.86
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0272571

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -270.0719910, 314.8731689, -286.4641724, 335.2057190, -605.2777100, 601.3373413
1: -223.6125946, 254.4744873, -238.0624084, 270.9075623, -494.5200806, 492.5368958
2: -180.4648132, 249.7426910, -191.7396545, 266.1152344, -446.5800171, 441.4823608
3: -252.6415710, 317.9813843, -269.0568237, 338.5891724, -591.2307129, 587.0382080
4: -232.0352020, 337.6802063, -246.5367889, 359.9270935, -591.9621582, 584.2169800

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331353, upper bound: 607.0331352
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331353, upper bound: 607.0331396
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -479.4232178, 560.6737061, -284.0694580, 332.0924377, -809.9136353, 844.4136353
1: -404.7547607, 452.2300110, -236.0105743, 268.4347229, -670.2072144, 688.1444092
2: -321.5619202, 446.7430420, -190.0851288, 263.6529541, -584.5635376, 636.2639160
3: -452.2616272, 565.2727051, -266.5987549, 335.5053101, -787.7669678, 830.5764160
4: -410.7044983, 606.4302979, -244.3632355, 356.5822754, -766.9104614, 849.6522217

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331396, upper bound: 607.0331353
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331396, upper bound: 607.0331396
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -270.0719910, 314.8731689, -456.6022034, 536.1311646, -805.6759033, 770.7769165
1: -223.6125946, 254.4744873, -385.3909607, 431.6862488, -655.2988281, 637.1831665
2: -180.4648132, 249.7426910, -305.7212830, 425.9559631, -605.9263306, 555.0603638
3: -252.6415710, 317.9813843, -430.5425720, 541.3033447, -793.0770264, 748.5238037
4: -232.0352020, 337.6802063, -391.2793579, 578.0347900, -809.3136597, 728.9595947

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300472, upper bound: 607.0341313
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300816, upper bound: 607.0358528
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -481.7822876, 563.9155273, -449.5650635, 528.3358765, -1006.7196045, 1011.1618652
1: -406.8581238, 454.8312073, -379.7332764, 425.4093933, -828.4018555, 830.6162720
2: -323.2179260, 449.4011230, -301.1425781, 419.8781128, -741.4274902, 749.0262451
3: -454.8097839, 568.5200806, -424.1198730, 533.3691406, -986.9890747, 990.9207764
4: -412.8977356, 610.0289917, -385.4289246, 569.7443848, -980.4860840, 993.2827148

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300557, upper bound: 607.0313260
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300957, upper bound: 607.0328398
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -453.6451721, 532.8193970, -265.5441589, 310.3986206, -763.3507690, 797.7982178
1: -383.0025024, 428.9759827, -220.7506409, 250.9053802, -631.2094727, 649.7266235
2: -303.7538757, 423.3137207, -177.6930847, 246.4173889, -549.7713013, 600.5130005
3: -427.7451477, 537.9436035, -249.0524139, 313.7485962, -741.4937744, 786.1229248
4: -388.7535400, 574.4205322, -228.3768921, 333.2054443, -721.9361572, 802.0381470

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0341471, upper bound: 607.0300612
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325389, upper bound: 607.0300649
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -453.2456970, 532.3560181, -382.1476440, 441.8602905, -894.0289917, 913.5317993
1: -382.6009216, 428.6413879, -317.0899353, 357.1013184, -736.7573853, 745.1353760
2: -303.4963684, 422.9486084, -254.7293854, 350.3905640, -653.3466187, 676.9282837
3: -427.4142151, 537.4869995, -355.7713013, 446.3111267, -873.7253418, 892.3572388
4: -388.4600525, 573.9308472, -326.7114868, 474.3641663, -862.5667114, 899.5466309

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0358197, upper bound: 607.0281114
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0358197, upper bound: 607.0301107
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -439.8607178, 516.9405518, -456.9133606, 537.2222290, -974.4500122, 971.2510986
1: -371.7370911, 416.0641479, -385.8175659, 432.4940796, -800.5106812, 798.2233887
2: -294.5429382, 410.6534424, -306.0419922, 426.8558044, -719.9204712, 715.1987305
3: -414.6834106, 521.9345093, -431.0969849, 542.2864990, -955.5986938, 951.6446533
4: -376.8948669, 557.2146606, -391.7742310, 579.1635132, -954.1837158, 947.0740967

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280958, upper bound: 607.0280958
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280958, upper bound: 607.0281045
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -559.7188110, 652.5390625, -456.5163269, 536.7534180, -1093.2166748, 1106.1293945
1: -470.3627930, 525.7763062, -385.4084473, 432.1513977, -897.9404907, 907.3399658
2: -373.9528809, 517.9055786, -305.7796936, 426.4836426, -798.6362305, 822.0823364
3: -524.8026733, 658.6452026, -430.7303772, 541.8178711, -1065.1838379, 1087.7246094
4: -478.5014343, 702.7322388, -391.4726257, 578.6552124, -1054.9881592, 1092.1110840

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0281045, upper bound: 607.0300584
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0281045, upper bound: 607.0301002
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -285.8565063, 334.4877930, -431.4479675, 498.9679565, -784.7072144, 764.0232544
1: -237.5498810, 270.3281555, -365.8478088, 402.8869019, -640.4037476, 631.9046631
2: -191.3265686, 265.5422974, -289.3208618, 396.3479614, -587.2045288, 553.7825317
3: -268.4792175, 337.8635559, -403.4279175, 506.2048340, -773.2265625, 741.2913818
4: -246.0056915, 359.1473083, -369.0462341, 537.3262939, -782.7354126, 727.2744751

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224973, upper bound: 607.0346538
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224674, upper bound: 607.0324874
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -286.4641724, 335.2057190, -444.0654602, 513.8726196, -800.2390137, 777.3029785
1: -238.0624084, 270.9075623, -376.4631348, 414.9239807, -652.9785767, 643.1866455
2: -191.7396545, 266.1152344, -297.9029846, 408.2196960, -599.5012207, 562.9459839
3: -269.0568237, 338.5891724, -415.4518433, 521.4194336, -789.0286865, 754.0410156
4: -246.5367889, 359.9270935, -380.1206665, 553.4392090, -799.4288330, 739.1279907

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265696, upper bound: 607.0348347
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265422, upper bound: 607.0326520
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -456.9133606, 537.2222290, -441.9322815, 513.0078735, -967.7342529, 975.1085205
1: -385.8175659, 432.4940796, -374.9511414, 414.1801147, -795.9898682, 802.2770996
2: -306.0419922, 426.8558044, -296.6209106, 407.9687500, -712.5588379, 721.2343140
3: -431.0969849, 542.2864990, -414.4992676, 520.0053101, -949.0933228, 955.3628540
4: -391.7742310, 579.1635132, -378.5418701, 552.9565430, -943.1152344, 954.7333374

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270358, upper bound: 607.0280486
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0270358, upper bound: 607.0299895
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -456.5163269, 536.7534180, -557.2149658, 644.2816162, -1098.6209717, 1090.1711426
1: -385.4084473, 432.1513977, -469.5955200, 520.2958374, -901.8195190, 896.3474121
2: -305.7796936, 426.4836426, -373.0922241, 511.8468323, -816.1819458, 797.3123169
3: -430.7303772, 541.8178711, -520.8032837, 652.1538086, -1080.9627686, 1061.0794678
4: -391.4726257, 578.6552124, -476.5472412, 693.9509888, -1083.5808105, 1052.3179932

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0286299, upper bound: 607.0280579
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0286299, upper bound: 607.0300330
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -271.7697449, 317.5826416, -514.0012207, 596.5606079, -866.3510742, 826.9982910
1: -225.7842102, 256.7298584, -441.5431824, 480.7114258, -705.4287109, 689.4071045
2: -181.8435516, 252.0915070, -344.8414001, 474.4323425, -654.7170410, 594.9528809
3: -254.8348236, 320.8648987, -482.4889526, 606.5974121, -858.0602417, 802.9031372
4: -233.7117615, 340.9127808, -438.5628052, 645.2090454, -876.4779053, 777.8692627

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268203, upper bound: 607.0348980
time: 0.96 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268275, upper bound: 607.0332360
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -384.5713806, 443.0445862, -515.0581055, 597.7996216, -978.7230225, 952.1746826
1: -315.3923950, 358.5366821, -442.6831360, 481.6140747, -795.3786011, 791.2180786
2: -256.0829163, 350.1005554, -345.5996094, 475.3015137, -728.8765259, 693.2478638
3: -354.6243591, 447.7526855, -483.4678955, 607.8311768, -958.5656738, 929.5932617
4: -328.9025269, 471.8538818, -439.4045410, 646.3851929, -971.4544678, 909.4688721

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268134, upper bound: 607.0327630
time: 0.93 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268134, upper bound: 607.0327630
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -446.3182983, 524.5844727, -550.2741089, 643.6594849, -1085.4537354, 1068.0136719
1: -377.1416321, 422.4203796, -472.7774963, 518.3488159, -890.1964111, 885.2538452
2: -298.9749756, 416.8713989, -369.9910278, 511.9668884, -808.2042847, 783.6143188
3: -421.0003967, 529.6852417, -519.6529541, 652.2393799, -1069.2341309, 1047.0458984
4: -382.6351318, 565.7437744, -470.9828796, 695.7875366, -1074.8300781, 1033.0159912

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0226323, upper bound: 607.0253957
time: 1.07 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0216695, upper bound: 607.0288844
time: 0.96 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268124, upper bound: 607.0290995
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -554.8208618, 648.2557373, -549.2106934, 642.3543701, -1191.6505127, 1189.8013916
1: -464.0573425, 522.7665405, -471.9148560, 517.2646484, -976.2656860, 984.0983276
2: -371.1120605, 513.9317017, -369.2695923, 510.8794556, -878.7612305, 879.7321777
3: -518.9692993, 654.4575806, -518.5298462, 650.8856812, -1165.5364990, 1170.0380859
4: -475.4354858, 695.7420044, -470.0264587, 694.2766113, -1165.3395996, 1162.0681152

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268073, upper bound: 607.0271905
time: 1.04 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268073, upper bound: 607.0271905
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -431.4391785, 498.9564209, -284.7556152, 333.1548767, -762.6819458, 783.5947876
1: -365.8389587, 402.8779602, -236.6221466, 269.2505798, -630.8178711, 639.4669189
2: -289.3144531, 396.3388367, -190.5757599, 264.4775085, -552.7116089, 586.4443359
3: -403.4181213, 506.1932068, -267.4127502, 336.5216980, -739.9396973, 772.1486816
4: -369.0381165, 537.3138428, -245.0337372, 357.7098083, -725.8293457, 781.7492065

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0346538, upper bound: 607.0224973
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0324874, upper bound: 607.0224674
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -444.0567627, 513.8611450, -285.3448181, 333.8539124, -775.9426880, 799.1080322
1: -376.4543762, 414.9150391, -237.1194763, 269.8140259, -642.0838013, 652.0263672
2: -297.8966370, 408.2106018, -190.9758759, 265.0350037, -561.8594360, 598.7277832
3: -415.4420776, 521.4078369, -267.9738159, 337.2276611, -752.6696167, 787.9345093
4: -380.1126404, 553.4266357, -245.5482330, 358.4687500, -737.6621704, 798.4259644

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0348347, upper bound: 607.0265696
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0326518, upper bound: 607.0265422
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -441.9322815, 513.0078735, -445.4366760, 523.2000122, -961.0624390, 956.2263184
1: -374.9511414, 414.1801147, -376.2823792, 421.1847534, -790.9564819, 786.4107056
2: -296.6209106, 407.9687500, -298.3050842, 415.7112122, -710.0789795, 704.7883301
3: -414.4992676, 520.0053101, -419.9384460, 528.2539673, -941.2951050, 937.9063721
4: -378.5418701, 552.9565430, -381.6887207, 564.1453857, -939.6945801, 932.9783325

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280486, upper bound: 607.0270358
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280486, upper bound: 607.0270422
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -557.2149658, 644.2816162, -445.1123352, 522.8053589, -1076.2008057, 1087.1866455
1: -469.5955200, 520.2958374, -375.9301147, 420.9009094, -885.0880127, 892.2982788
2: -373.0922241, 511.8468323, -298.0887451, 415.3949890, -786.2128296, 808.4617310
3: -520.8032837, 652.1538086, -419.6341248, 527.8610229, -1047.0883789, 1069.8389893
4: -476.5472412, 693.9509888, -381.4495544, 563.7119751, -1037.3543701, 1073.5084229

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280579, upper bound: 607.0286299
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280579, upper bound: 607.0286490
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -513.9898682, 596.5445557, -270.7373657, 316.3435669, -825.7476807, 865.3030396
1: -441.5350647, 480.6980896, -224.9148560, 255.7257538, -688.3936157, 704.5466919
2: -344.8338013, 474.4212341, -181.1396484, 251.1015167, -593.9553833, 654.0015259
3: -482.4775391, 606.5831909, -253.8401947, 319.6137085, -801.6400757, 857.0507812
4: -438.5520935, 645.1956177, -232.8022766, 339.5761719, -776.5232544, 875.5527344

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0348980, upper bound: 607.0268203
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0332360, upper bound: 607.0268277
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -515.0343018, 597.7675781, -383.6549683, 441.9339294, -951.0377197, 977.7703247
1: -442.6642151, 481.5881348, -314.6208801, 357.6394348, -790.2996216, 794.5796509
2: -345.5831299, 475.2771912, -255.4591370, 349.2123108, -692.3422241, 728.2266235
3: -483.4434814, 607.8010254, -353.7373962, 446.6395874, -928.4533691, 957.6463623
4: -439.3823242, 646.3537598, -328.0957642, 470.6525574, -908.2451172, 970.6122437

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327631, upper bound: 607.0268134
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327631, upper bound: 607.0268134
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -550.8312988, 644.3563843, -434.8822021, 510.5615234, -1054.5289307, 1074.6743164
1: -473.2405396, 518.9094238, -367.6223145, 411.1117249, -874.4083252, 881.1909790
2: -370.3688354, 512.5308228, -291.2521667, 405.7142334, -772.8263550, 801.0115967
3: -520.2126465, 652.9137573, -409.8415222, 515.6508179, -1033.5311279, 1058.7233887
4: -471.4726257, 696.5507202, -372.5696411, 550.7170410, -1018.4577026, 1065.4812012

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0254641, upper bound: 607.0226325
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0289052, upper bound: 607.0216695
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0291704, upper bound: 607.0268131
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -549.8206787, 643.1312256, -543.8267822, 634.5370483, -1176.6402588, 1181.3590088
1: -472.4326477, 517.8869629, -454.8020020, 511.7872620, -973.5929565, 967.5753784
2: -369.6881714, 511.5083313, -363.7036438, 503.0183716, -869.2170410, 871.9216919
3: -519.1533813, 651.6406250, -508.1711426, 640.8057861, -1156.9497070, 1155.4482422
4: -470.5679016, 695.1279907, -465.7715149, 681.0022583, -1147.8437500, 1156.4395752

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273073, upper bound: 607.0268076
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273072, upper bound: 607.0268076
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -461.9153442, 537.3094482, -464.9393005, 541.0213623, -999.4320068, 998.7866821
1: -391.6099548, 433.7934570, -394.1322937, 436.8084106, -823.0317383, 822.5768433
2: -310.1578979, 427.3957214, -312.2680054, 430.4052429, -738.4300537, 737.5233154
3: -434.1242371, 544.3061523, -437.1111755, 547.9950562, -980.1292114, 979.3800659
4: -396.1182556, 579.3615723, -398.8383484, 583.4384766, -976.9216919, 975.5650635

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0233074
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0233075
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -568.1647339, 665.9223022, -461.9153442, 537.3094482, -1099.0634766, 1122.0999756
1: -487.8350830, 536.2718506, -391.6099548, 433.7934570, -911.4801636, 921.1367188
2: -382.2286987, 529.9315796, -310.1578979, 427.3957214, -806.4324951, 836.7152710
3: -537.6182861, 674.1225586, -434.1242371, 544.3061523, -1079.1173096, 1104.1781006
4: -486.8578186, 720.0869141, -396.1182556, 579.3615723, -1062.7009277, 1111.6333008

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273235, upper bound: 607.0235714
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0285733, upper bound: 607.0235760
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -557.5258179, 653.3888550, -535.8256226, 624.1657715, -1173.6717529, 1181.3089600
1: -479.0955200, 526.1940918, -456.5402527, 503.6176758, -971.3153687, 973.5148315
2: -375.1179504, 520.0421753, -360.2144775, 496.8298035, -867.8972778, 876.0298462
3: -527.6604004, 661.5354614, -504.7485657, 631.6193237, -1154.8897705, 1161.8178711
4: -477.6614990, 706.7284546, -459.1738281, 674.1534424, -1147.0148926, 1160.2453613

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224015, upper bound: 607.0268670
time: 1.06 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0232851, upper bound: 607.0269547
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -444.8146973, 516.7648926, -563.6619873, 660.4889526, -1099.4301758, 1073.9577637
1: -377.4189148, 417.2292786, -484.0943298, 531.8816528, -902.4838257, 891.0837402
2: -298.6468811, 411.0758667, -379.1786194, 525.5844116, -820.7806396, 787.0260620
3: -417.5617371, 523.6961670, -533.2381592, 668.6888428, -1082.1572266, 1054.0660400
4: -381.1652222, 557.1831665, -482.8868713, 714.2012939, -1090.6787109, 1036.5471191

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275703, upper bound: 607.0270343
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275704, upper bound: 607.0270403
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -560.4065552, 648.3799438, -563.0426025, 659.7294312, -1214.5053711, 1204.9532471
1: -472.3208923, 523.6036987, -483.4830933, 531.3002319, -996.5598755, 996.9695435
2: -375.3253174, 515.2205200, -378.7529297, 524.9683228, -896.8184814, 890.7507324
3: -524.1104126, 656.1795654, -532.6375122, 667.9337769, -1187.8258057, 1186.0205078
4: -479.4255371, 698.5368042, -482.3663025, 713.3538818, -1188.1695557, 1177.1452637

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275804, upper bound: 607.0286288
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275804, upper bound: 607.0286472
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -567.6685791, 665.3579712, -570.2347412, 668.6093140, -1227.5606689, 1226.9362793
1: -487.4307556, 535.7752686, -489.6099243, 538.4390869, -1014.2953491, 1013.8818970
2: -381.8864746, 529.4561157, -383.6712646, 532.1336670, -909.5463867, 908.6873779
3: -537.2376099, 673.4689331, -539.7968140, 676.7504883, -1209.1306152, 1208.3653564
4: -486.3699646, 719.4578247, -488.7454834, 723.0624390, -1203.9730225, 1202.7132568

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273239, upper bound: 607.0272569
time: 1.03 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273239, upper bound: 607.0272567
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -630.6661987, 739.7068481, -557.7584229, 653.6845703, -1272.8811035, 1287.1252441
1: -543.3923340, 595.5401001, -479.2927246, 526.4332275, -1055.4472656, 1061.9548340
2: -424.9387817, 589.1366577, -375.2799683, 520.2851562, -939.4981079, 958.9763184
3: -598.2095947, 748.1354370, -527.9039917, 661.8233032, -1254.5811768, 1269.3891602
4: -540.1110840, 801.3413086, -477.8716736, 707.0584106, -1240.5034180, 1272.1657715

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273288, upper bound: 607.0272567
time: 1.44 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0273288, upper bound: 607.0272571
time: 0.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.59 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0331353, upper bound: 607.0331352
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0331353, upper bound: 607.0331396
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0331396, upper bound: 607.0331353
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0331396, upper bound: 607.0331396
NS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0300472, upper bound: 607.0341313
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0300816, upper bound: 607.0358528
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0300557, upper bound: 607.0313260
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0300957, upper bound: 607.0328398
NS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0341471, upper bound: 607.0300612
NS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0325389, upper bound: 607.0300649
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0358197, upper bound: 607.0281114
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0358197, upper bound: 607.0301107
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0280958, upper bound: 607.0280958
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0280958, upper bound: 607.0281045
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0281045, upper bound: 607.0300584
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0281045, upper bound: 607.0301002
NS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0224973, upper bound: 607.0346538
NS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0224674, upper bound: 607.0324874
NS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0265696, upper bound: 607.0348347
NS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0265422, upper bound: 607.0326520
NS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0270358, upper bound: 607.0280486
NS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0270358, upper bound: 607.0299895
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0286299, upper bound: 607.0280579
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0286299, upper bound: 607.0300330
NS_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268203, upper bound: 607.0348980
NS_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268275, upper bound: 607.0332360
NS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268134, upper bound: 607.0327630
NS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268134, upper bound: 607.0327630
NS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0216695, upper bound: 607.0288844
NS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268124, upper bound: 607.0290995
NS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268073, upper bound: 607.0271905
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0268073, upper bound: 607.0271905
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0346538, upper bound: 607.0224973
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0324874, upper bound: 607.0224674
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0348347, upper bound: 607.0265696
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0326518, upper bound: 607.0265422
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0280486, upper bound: 607.0270358
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0280486, upper bound: 607.0270422
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0280579, upper bound: 607.0286299
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0280579, upper bound: 607.0286490
NS_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0348980, upper bound: 607.0268203
NS_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0332360, upper bound: 607.0268277
NS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0327631, upper bound: 607.0268134
NS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0327631, upper bound: 607.0268134
NS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0289052, upper bound: 607.0216695
NS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0291704, upper bound: 607.0268131
NS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273073, upper bound: 607.0268076
NS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273072, upper bound: 607.0268076
NS_A2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0233074
NS_A2_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0233309, upper bound: 607.0233075
NS_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273235, upper bound: 607.0235714
NS_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0285733, upper bound: 607.0235760
NS_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0224015, upper bound: 607.0268670
NS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0232851, upper bound: 607.0269547
NS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0275703, upper bound: 607.0270343
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0275704, upper bound: 607.0270403
NS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0275804, upper bound: 607.0286288
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0275804, upper bound: 607.0286472
NS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273239, upper bound: 607.0272569
NS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273239, upper bound: 607.0272567
NS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273288, upper bound: 607.0272567
NS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -607.0273288, upper bound: 607.0272571

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -270.0719910, 314.8731689, -270.0719910, 314.8731689, -584.9451294, 584.9451294
1: -223.6125946, 254.4744873, -223.6125946, 254.4744873, -478.0870361, 478.0870361
2: -180.4648132, 249.7426910, -180.4648132, 249.7426910, -430.2075195, 430.2075195
3: -252.6415710, 317.9813843, -252.6415710, 317.9813843, -570.6229248, 570.6229248
4: -232.0352020, 337.6802063, -232.0352020, 337.6802063, -569.7153931, 569.7153931

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327966, upper bound: 607.0345681
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0327966, upper bound: 607.0362076
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -270.0719910, 314.8731689, -478.4171448, 559.2822266, -829.0294800, 791.6622925
1: -223.6125946, 254.4744873, -403.8392334, 451.1198730, -674.6026001, 655.3190308
2: -180.4648132, 249.7426910, -320.8453979, 445.6054688, -625.4989624, 569.9294434
3: -252.6415710, 317.9813843, -451.1596375, 563.8817749, -815.2113037, 769.1409912
4: -232.0352020, 337.6802063, -409.7626038, 604.8861694, -835.7753296, 747.0741577

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331195, upper bound: 607.0366517
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331361, upper bound: 607.0365140
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -478.4108887, 559.2737427, -268.9600220, 313.5357666, -790.3192139, 827.9090576
1: -403.8330078, 451.1132812, -222.6748810, 253.3898010, -654.2276611, 673.6581421
2: -320.8407593, 445.5986328, -179.7055664, 248.6727295, -568.8549805, 624.7323608
3: -451.1524658, 563.8732910, -251.5680542, 316.6301575, -767.7825928, 814.1295776
4: -409.7567444, 604.8767090, -231.0514526, 336.2368774, -745.6252441, 834.7805176

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331368, upper bound: 607.0331185
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331368, upper bound: 607.0331345
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -481.7822876, 563.9155273, -481.7822876, 563.9155273, -1042.4726562, 1042.4726562
1: -406.8581238, 454.8312073, -406.8581238, 454.8312073, -857.4515381, 857.4515381
2: -323.2179260, 449.4011230, -323.2179260, 449.4011230, -770.8695679, 770.8695679
3: -454.8097839, 568.5200806, -454.8097839, 568.5200806, -1021.7062988, 1021.7062988
4: -412.8977356, 610.0289917, -412.8977356, 610.0289917, -1020.3634644, 1020.3634644

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331226, upper bound: 607.0331345
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0331368, upper bound: 607.0331345
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -249.3863220, 290.3250427, -452.1828003, 530.8461304, -779.6666870, 741.8014526
1: -206.4957275, 234.6801910, -381.7190552, 427.4079895, -633.9036865, 613.6923828
2: -166.5820618, 230.2462158, -302.7264404, 421.7317200, -587.8099976, 532.5673218
3: -232.8464661, 293.4009399, -426.2492371, 535.9943237, -767.9619141, 719.6501465
4: -214.0746307, 311.2335510, -387.4009399, 572.3038940, -785.6051636, 698.6325073

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300315, upper bound: 607.0341266
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300374, upper bound: 607.0325233
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -365.5653687, 421.1904297, -451.9660645, 530.6410522, -895.1882935, 872.0412598
1: -302.3995361, 340.4073181, -381.5026245, 427.2734375, -729.0568848, 718.9313354
2: -243.2815552, 333.7402344, -302.6042175, 421.5719604, -664.0738525, 635.8023682
3: -339.0806274, 425.3659973, -426.1228333, 535.7976074, -873.9627686, 851.4888306
4: -312.0025635, 451.7153625, -387.2824097, 572.0925903, -882.9005737, 838.7636108

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280781, upper bound: 607.0358116
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280781, upper bound: 607.0358528
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -459.7984009, 537.7586060, -445.1419678, 523.0457153, -979.3552246, 980.5357056
1: -388.5707703, 433.7257385, -376.0639954, 421.1234741, -805.7729492, 805.7858276
2: -308.4191895, 428.5906677, -298.1464233, 415.6475830, -722.3432617, 725.2033691
3: -433.6254272, 542.2889404, -419.8238525, 528.0578613, -960.4951782, 960.3410034
4: -393.6908569, 581.7775269, -381.5484009, 564.0065308, -955.5015869, 961.1416016

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0300283, upper bound: 607.0313231
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0286575, upper bound: 607.0308397
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -569.4069214, 661.6353760, -444.5174561, 522.3325195, -1088.9003906, 1103.9899902
1: -478.8410034, 533.9137573, -375.4831238, 420.5918884, -895.4097900, 905.6797485
2: -380.9854126, 526.7255859, -297.7411499, 415.0888672, -794.5411377, 822.9566040
3: -534.4058228, 667.2781982, -419.2919922, 527.3553467, -1060.6811523, 1085.0528564
4: -486.5560303, 714.9441528, -381.0646667, 563.2549438, -1048.0507812, 1093.7962646

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280917, upper bound: 607.0328006
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0280917, upper bound: 607.0328398
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -452.9662781, 531.9268799, -257.9457703, 300.7846680, -753.0373535, 789.3314209
1: -382.4202271, 428.2525330, -214.3491364, 243.0965271, -622.8017578, 642.6015015
2: -303.2771912, 422.5848389, -172.4906616, 238.6143494, -541.4943237, 594.5715332
3: -427.0494995, 537.0518188, -241.3327179, 304.1046143, -731.1541138, 777.5029297
4: -388.1293030, 573.4383545, -221.5868073, 322.6164246, -710.7296143, 794.2698364

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0341457, upper bound: 607.0281050
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0341457, upper bound: 607.0300612
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -451.6626587, 530.8749390, -328.8097229, 386.1489868, -836.3115845, 858.0640259
1: -381.4211121, 427.3666992, -277.2524414, 311.3551636, -689.6332397, 703.0148315
2: -302.4830627, 421.7815247, -220.6959534, 307.0518188, -608.6621704, 641.4713745
3: -426.0405884, 535.9138184, -309.8092346, 390.3789368, -815.8181152, 844.8762207
4: -387.1759033, 572.3118896, -282.5450134, 416.3402100, -802.5933838, 853.6417236

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325338, upper bound: 607.0281164
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0325338, upper bound: 607.0300649
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -436.8053589, 512.7673340, -380.0960999, 439.4405823, -875.1256714, 891.8521729
1: -369.0678406, 412.7216797, -315.3474426, 355.1840210, -721.1898804, 727.4746704
2: -292.3852539, 407.2892151, -253.3861084, 348.4746399, -640.3164673, 659.8889771
3: -411.4671021, 517.7991333, -353.8627625, 443.9015198, -855.3641357, 870.7152710
4: -374.0366211, 552.6964722, -325.0052490, 471.7327576, -845.5039673, 876.5231323

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0337497, upper bound: 607.0281016
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0320385, upper bound: 607.0281072
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -556.2728882, 647.9678955, -382.1476440, 441.8602905, -996.3745117, 1028.7789307
1: -467.4185486, 522.1068726, -317.0899353, 357.1013184, -820.5967407, 838.3593750
2: -371.5511780, 514.2301636, -254.7293854, 350.3905640, -721.0685425, 768.0690308
3: -521.3081665, 654.1291504, -355.7713013, 446.3111267, -967.5393677, 1008.6641235
4: -475.3287659, 697.8117065, -326.7114868, 474.3641663, -949.1016235, 1023.1576538

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0337497, upper bound: 607.0300997
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0320385, upper bound: 607.0301037
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -439.8607178, 516.9405518, -439.8607178, 516.9405518, -954.1306152, 954.1306152
1: -371.7370911, 416.0641479, -371.7370911, 416.0641479, -784.0507202, 784.0507202
2: -294.5429382, 410.6534424, -294.5429382, 410.6534424, -703.6892090, 703.6892090
3: -414.6834106, 521.9345093, -414.6834106, 521.9345093, -935.2013550, 935.2014160
4: -376.8948669, 557.2146606, -376.8948669, 557.2146606, -932.1708984, 932.1708984

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0281110, upper bound: 607.0284920
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0276450, upper bound: 607.0281356
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -439.8607178, 516.9405518, -556.6879883, 649.0161743, -1085.9133301, 1070.3410645
1: -371.7370911, 416.0641479, -467.7961121, 522.9682617, -890.7506104, 879.3053589
2: -294.5429382, 410.6534424, -371.9503479, 515.1296387, -808.0722046, 780.7738037
3: -414.6834106, 521.9345093, -521.9807129, 655.1315918, -1068.1232910, 1042.4439697
4: -376.8948669, 557.2146606, -475.9447021, 698.9423828, -1073.7375488, 1030.9245605

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0281110, upper bound: 607.0284920
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0276450, upper bound: 607.0281356
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -556.7445679, 649.0763550, -439.8607178, 516.9405518, -1070.3978271, 1085.9736328
1: -467.8433838, 523.0175781, -371.7370911, 416.0641479, -879.3515625, 890.7997437
2: -371.9877625, 515.1770020, -294.5429382, 410.6534424, -780.8110352, 808.1195068
3: -522.0308228, 655.1932983, -414.6834106, 521.9345093, -1042.4942627, 1068.1850586
4: -475.9918518, 699.0064087, -376.8948669, 557.2146606, -1030.9716797, 1073.8016357

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0276479, upper bound: 607.0299932
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0276172, upper bound: 607.0286555
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -559.7188110, 652.5390625, -559.7188110, 652.5390625, -1208.6365967, 1208.6365967
1: -470.3627930, 525.7763062, -470.3627930, 525.7763062, -991.3181763, 991.3181763
2: -373.9528809, 517.9055786, -373.9528809, 517.9055786, -889.9175415, 889.9175415
3: -524.8026733, 658.6452026, -524.8026733, 658.6452026, -1181.6774902, 1181.6774902
4: -478.5014343, 702.7322388, -478.5014343, 702.7322388, -1178.7940674, 1178.7940674

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0276479, upper bound: 607.0299930
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0276172, upper bound: 607.0286591
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -271.1522522, 316.8530579, -424.7827148, 491.0576477, -761.9984131, 739.6219482
1: -225.2626343, 256.1412659, -360.3225098, 396.5168457, -621.7372437, 612.1455688
2: -181.4235992, 251.5096283, -284.8180237, 390.0506287, -570.9959106, 535.2282104
3: -254.2471466, 320.1276855, -397.0643921, 498.2646790, -750.9973145, 717.1920776
4: -233.1720276, 340.1217346, -363.2306824, 528.8227539, -761.3709106, 702.4282837

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224772, upper bound: 607.0346403
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224842, upper bound: 607.0329291
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -383.9788208, 442.3482666, -423.7045593, 489.8214722, -872.0833740, 862.8494873
1: -314.8902283, 357.9738159, -359.5733337, 395.4899292, -709.8399048, 712.2730713
2: -255.6806335, 349.5448303, -284.1543579, 389.0682983, -643.3831787, 632.2088013
3: -354.0639038, 447.0486450, -396.1040955, 497.0357056, -849.1801758, 842.3234253
4: -328.3866577, 471.0994263, -362.3570557, 527.4387207, -853.9279785, 832.3925171

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224674, upper bound: 607.0324874
time: 0.94 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224674, upper bound: 607.0324874
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -271.7697449, 317.5826416, -437.4911804, 506.1122437, -777.6919556, 753.0064697
1: -225.7842102, 256.7298584, -371.0161743, 408.6751099, -634.4435425, 623.5167236
2: -181.8435516, 252.0915070, -293.4691162, 402.0503235, -583.4257812, 544.4717407
3: -254.8348236, 320.8648987, -409.1988220, 513.6317749, -766.9620361, 730.0637207
4: -233.7117615, 340.9127808, -374.4005737, 545.1001587, -778.2401733, 714.3917847

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265485, upper bound: 607.0348325
time: 1.09 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265557, upper bound: 607.0331521
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -384.5713806, 443.0445862, -436.0961304, 504.4742126, -887.3480835, 875.8748169
1: -315.3923950, 358.5366821, -369.9880066, 407.3309326, -722.2062378, 723.3385620
2: -256.0829163, 350.1005554, -292.5774536, 400.7385559, -655.4707031, 641.1967773
3: -354.6243591, 447.7526855, -407.9012146, 512.0021973, -864.7168579, 854.8184204
4: -328.9025269, 471.8538818, -373.2264099, 543.2722168, -870.3311768, 844.0130615

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0224674, upper bound: 607.0326518
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265422, upper bound: 607.0326518
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -439.8607178, 516.9405518, -441.9002075, 512.9669800, -950.5729370, 954.7570190
1: -371.7370911, 416.0641479, -374.9238281, 414.1470947, -781.7842407, 785.7894897
2: -294.5429382, 410.6534424, -296.5986633, 407.9350891, -701.0157471, 704.9807739
3: -414.6834106, 521.9345093, -414.4655457, 519.9655151, -932.6102295, 934.9319458
4: -376.8948669, 557.2146606, -378.5133972, 552.9105225, -928.1660767, 932.6920166

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275285, upper bound: 607.0280729
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266727, upper bound: 607.0276003
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -555.6116333, 647.8784180, -442.0117188, 513.1095581, -1065.8529053, 1085.5148926
1: -466.9008484, 522.0381470, -375.0185852, 414.2620239, -876.2747803, 891.6582031
2: -371.2403870, 514.2365723, -296.6761780, 408.0522766, -777.5088501, 808.5483398
3: -521.0317993, 653.9650269, -414.5827637, 520.1042480, -1039.0423584, 1066.8050537
4: -475.0493469, 697.7355957, -378.6128845, 553.0701904, -1026.1861572, 1073.1492920

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0275285, upper bound: 607.0299625
time: 0.98 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0266727, upper bound: 607.0285974
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -439.8607178, 516.9405518, -554.7429199, 641.3614502, -1079.0056152, 1067.8511963
1: -371.7370911, 416.0641479, -467.5134888, 517.9688110, -885.7073364, 878.1713867
2: -294.5429382, 410.6534424, -371.4646301, 509.5313416, -802.6285400, 779.8203735
3: -414.6834106, 521.9345093, -518.4681396, 649.2590942, -1061.9764404, 1038.8254395
4: -376.8948669, 557.2146606, -474.4627380, 690.7805786, -1065.8190918, 1028.7215576

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265033, upper bound: 607.0275965
time: 1.23 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0257181, upper bound: 607.0275450
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -559.7188110, 652.5390625, -557.4854126, 644.5669556, -1201.4094238, 1205.8522949
1: -470.3627930, 525.7763062, -469.8086853, 520.5270996, -986.0258179, 989.9360352
2: -373.9528809, 517.9055786, -373.2688293, 512.0710449, -884.2403564, 888.7653809
3: -524.8026733, 658.6452026, -521.0386353, 652.4401855, -1175.1976318, 1177.8078613
4: -478.5014343, 702.7322388, -476.7705383, 694.2526855, -1170.5651855, 1176.3403320

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0257415, upper bound: 607.0299110
time: 0.91 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0257181, upper bound: 607.0285857
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -264.1855164, 308.0421448, -510.8789978, 592.6516113, -854.9017944, 814.3451538
1: -219.3764648, 248.9773407, -438.9122314, 477.5882568, -695.9414673, 679.0122070
2: -176.6547546, 244.3464508, -342.7180786, 471.3461304, -646.4462891, 585.0914917
3: -247.1581421, 311.2823181, -479.4033813, 602.7692871, -846.5578613, 790.2340698
4: -226.9504852, 330.4039612, -435.8074951, 641.0576172, -865.5720825, 764.6384277

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_A1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0258670, upper bound: 607.0347510
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267368, upper bound: 607.0347584
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -334.4300842, 392.7620850, -525.7376099, 612.4284058, -943.5728149, 912.9337158
1: -281.8948364, 316.7151794, -451.8100281, 493.2639465, -772.0552979, 759.1514893
2: -224.4830933, 312.2936401, -353.0484314, 486.9616089, -709.2882080, 662.7899780
3: -315.2000732, 396.9967651, -494.8081055, 621.7568359, -933.5366821, 890.3164062
4: -287.4388733, 423.5255127, -449.1714478, 661.9973145, -946.5245361, 870.0618896

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A1_A1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0264869, upper bound: 607.0300337
time: 0.93 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0265120, upper bound: 607.0332118
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -384.5713806, 443.0445862, -505.9983521, 587.1556396, -968.1179199, 943.0098877
1: -315.3923950, 358.5366821, -435.3005066, 473.1302185, -786.9168701, 783.8057861
2: -256.0829163, 350.1005554, -339.5669556, 467.0083618, -720.6018677, 687.1761475
3: -354.6243591, 447.7526855, -475.0687561, 597.2797241, -947.9793701, 921.2109985
4: -328.9025269, 471.8538818, -431.5823059, 635.2769775, -960.3366699, 901.6774902

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267950, upper bound: 607.0327610
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268030, upper bound: 607.0314600
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -384.5713806, 443.0445862, -609.5689697, 706.2601929, -1086.7222900, 1046.2861328
1: -315.3923950, 358.5366821, -518.4563599, 569.6968384, -883.2826538, 867.7442627
2: -256.0829163, 350.1005554, -408.5822449, 560.7944946, -814.3309937, 756.0357056
3: -354.6243591, 447.7526855, -569.4840698, 717.1520386, -1067.6187744, 1015.6538086
4: -328.9025269, 471.8538818, -520.5573120, 760.9614258, -1086.1773682, 990.1573486

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267950, upper bound: 607.0327610
time: 0.94 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0268030, upper bound: 607.0314599
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -445.6710815, 523.7940063, -538.2094116, 629.0985107, -1070.1771240, 1055.1087646
1: -376.5963440, 421.7852783, -462.6865845, 506.6259766, -877.8796387, 874.3399658
2: -298.5318298, 416.2423706, -361.7415161, 500.3984680, -796.1491699, 774.7142944
3: -420.3696594, 528.8895874, -507.9712830, 637.4703369, -1053.7984619, 1034.5734863
4: -382.0598755, 564.8876343, -460.2583313, 680.0733032, -1058.4545898, 1021.3922729

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0216576, upper bound: 607.0288844
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0216444, upper bound: 607.0286732
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -446.3182983, 524.5844727, -548.7794189, 641.8664551, -1083.6431885, 1066.4884033
1: -377.1416321, 422.4203796, -471.5314331, 516.8981934, -888.7329102, 883.9907837
2: -298.9749756, 416.8713989, -368.9856262, 510.5308228, -806.7619629, 782.5935059
3: -421.0003967, 529.6852417, -518.2261353, 650.4357300, -1067.4123535, 1045.6104736
4: -382.6351318, 565.7437744, -469.6701965, 693.8477783, -1072.8798828, 1031.6961670

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0258552, upper bound: 607.0289738
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267280, upper bound: 607.0289741
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -554.8208618, 648.2557373, -540.1079712, 631.6431885, -1180.9816895, 1180.5970459
1: -464.0573425, 522.7665405, -464.5009460, 508.6923828, -967.6970215, 976.6572876
2: -371.1120605, 513.9317017, -363.2038269, 502.4903564, -870.3897095, 873.6249390
3: -518.9692993, 654.4575806, -510.0678711, 640.2658691, -1154.8856201, 1161.5963135
4: -475.4354858, 695.7420044, -462.1533508, 683.0494995, -1154.1048584, 1154.2373047

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0258431, upper bound: 607.0271102
time: 1.29 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267248, upper bound: 607.0271102
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -554.8208618, 648.2557373, -644.9721069, 751.5343628, -1300.3540039, 1285.0944824
1: -464.0573425, 522.7665405, -548.4967041, 605.9374390, -1064.7318115, 1061.3828125
2: -371.1120605, 513.9317017, -432.7437744, 596.8822021, -964.7021484, 943.0866089
3: -518.9692993, 654.4575806, -605.1091309, 761.0830078, -1275.4409180, 1256.6470947
4: -475.4354858, 695.7420044, -551.9665527, 809.5656128, -1280.7457275, 1243.4929199

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0258431, upper bound: 607.0271102
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -607.0267248, upper bound: 607.0271102
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -424.7745056, 491.0467224, -270.1350098, 315.6341248, -738.3945923, 760.9701538
1: -360.3141785, 396.5083618, -224.4064026, 255.1532898, -611.1488647, 620.8723755
2: -284.8120728, 390.0418701, -180.7304993, 250.5358429, -534.2481689, 570.2937012
3: -397.0551453, 498.2536926, -253.2684326, 318.8961182, -715.9511719, 750.0075073
4: -363.2229309, 528.8108521, -232.2766571, 338.8075256, -701.1067505, 760.4619141

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.32 + 415.77 = 420.08 seconds
