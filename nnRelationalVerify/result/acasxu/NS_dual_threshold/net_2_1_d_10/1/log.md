## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 1541.9334605111999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-791.8958740, 987.6563721, -791.8958740, 987.6563721, -1779.5522461, 1779.5522461)
1: (-576.4523926, 773.4627075, -576.4523926, 773.4627075, -1349.9150391, 1349.9150391)
2: (-492.7531433, 765.4989014, -492.7531433, 765.4989014, -1258.2518311, 1258.2518311)
3: (-691.0799561, 926.5178833, -691.0799561, 926.5178833, -1617.5975342, 1617.5975342)
4: (-652.9229736, 1029.5030518, -652.9229736, 1029.5030518, -1682.4260254, 1682.4260254)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.28 + 2.12 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1541.9488800, upper bound: 1541.9488800

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9463602, upper bound: 1541.9476119
time: 0.89 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9463601, upper bound: 1541.9479269
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.76 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -1541.9463602, upper bound: 1541.9476119
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -1541.9463601, upper bound: 1541.9479269

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -785.8185425, 980.1410522, -787.9378052, 984.4304810, -1770.2490234, 1768.0788574
1: -572.0701904, 767.5969238, -576.4047852, 772.3870239, -1344.4572754, 1344.0017090
2: -488.9977722, 759.6923828, -492.4450684, 764.5554199, -1253.5532227, 1252.1373291
3: -685.8312378, 919.5076294, -690.5776978, 926.0027466, -1611.8339844, 1610.0853271
4: -647.9619751, 1021.6812744, -652.7169800, 1027.9724121, -1675.9343262, 1674.3981934

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9462902, upper bound: 1541.9462902
time: 0.91 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9462902, upper bound: 1541.9476119
time: 0.74 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -787.0170288, 981.4689331, -763.6300659, 951.8750000, -1738.8920898, 1745.0989990
1: -572.8715210, 768.6293945, -555.7418823, 745.5239258, -1318.3955078, 1324.3713379
2: -489.6823730, 760.6801147, -474.9835510, 737.6679077, -1227.3503418, 1235.6636963
3: -686.7753906, 920.7142944, -666.1837769, 892.9794312, -1579.7542725, 1586.8979492
4: -648.8524780, 1023.0161743, -629.3654175, 992.0773926, -1640.9299316, 1652.3814697

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9476119, upper bound: 1541.9463601
time: 0.79 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9476119, upper bound: 1541.9479269
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.92 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 0, lower bound: -1541.9462902, upper bound: 1541.9462902
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 0, lower bound: -1541.9462902, upper bound: 1541.9476119
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 0, lower bound: -1541.9476119, upper bound: 1541.9463601
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 0, lower bound: -1541.9476119, upper bound: 1541.9479269

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -787.9378052, 984.4304810, -787.9378052, 984.4304810, -1772.3682861, 1772.3682861
1: -576.4047852, 772.3870239, -576.4047852, 772.3870239, -1348.7917480, 1348.7917480
2: -492.4450684, 764.5554199, -492.4450684, 764.5554199, -1257.0004883, 1257.0004883
3: -690.5776978, 926.0027466, -690.5776978, 926.0027466, -1616.5804443, 1616.5804443
4: -652.7169800, 1027.9724121, -652.7169800, 1027.9724121, -1680.6894531, 1680.6894531

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9461902, upper bound: 1541.9451923
time: 1.03 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9461992
time: 0.75 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -763.6300659, 951.8750000, -787.9378052, 984.4304810, -1748.0605469, 1739.8127441
1: -555.7418823, 745.5239258, -576.4047852, 772.3870239, -1328.1289062, 1321.9287109
2: -474.9835510, 737.6679077, -492.4450684, 764.5554199, -1239.5389404, 1230.1129150
3: -666.1837769, 892.9794312, -690.5776978, 926.0027466, -1592.1865234, 1583.5567627
4: -629.3654175, 992.0773926, -652.7169800, 1027.9724121, -1657.3378906, 1644.7944336

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9451923, upper bound: 1541.9461902
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9475409
time: 0.78 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -787.9378052, 984.4304810, -763.6300659, 951.8750000, -1739.8127441, 1748.0605469
1: -576.4047852, 772.3870239, -555.7418823, 745.5239258, -1321.9287109, 1328.1289062
2: -492.4450684, 764.5554199, -474.9835510, 737.6679077, -1230.1129150, 1239.5389404
3: -690.5776978, 926.0027466, -666.1837769, 892.9794312, -1583.5567627, 1592.1865234
4: -652.7169800, 1027.9724121, -629.3654175, 992.0773926, -1644.7944336, 1657.3378906

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9461902, upper bound: 1541.9452494
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9462690
time: 0.98 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -763.6300659, 951.8750000, -763.6300659, 951.8750000, -1715.5051270, 1715.5051270
1: -555.7418823, 745.5239258, -555.7418823, 745.5239258, -1301.2658691, 1301.2658691
2: -474.9835510, 737.6679077, -474.9835510, 737.6679077, -1212.6514893, 1212.6514893
3: -666.1837769, 892.9794312, -666.1837769, 892.9794312, -1559.1628418, 1559.1628418
4: -629.3654175, 992.0773926, -629.3654175, 992.0773926, -1621.4428711, 1621.4427490

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9451923, upper bound: 1541.9478269
time: 0.78 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9478294
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.88 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9461902, upper bound: 1541.9451923
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9461992
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9451923, upper bound: 1541.9461902
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9475409
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9461902, upper bound: 1541.9452494
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9462690
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9451923, upper bound: 1541.9478269
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -1541.9461992, upper bound: 1541.9478294

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -784.7306519, 980.4318237, -787.9378052, 984.4304810, -1769.1608887, 1768.3696289
1: -574.2277832, 769.3301392, -576.4047852, 772.3870239, -1346.6143799, 1345.7348633
2: -490.5498962, 761.5339966, -492.4450684, 764.5554199, -1255.1052246, 1253.9790039
3: -687.8717651, 922.4364624, -690.5776978, 926.0027466, -1613.8745117, 1613.0141602
4: -650.2122803, 1023.8786011, -652.7169800, 1027.9724121, -1678.1846924, 1676.5955811

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9408054, upper bound: 1541.9341846
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
time: 1.16 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -785.9101562, 981.8997803, -787.9378052, 984.4304810, -1770.3405762, 1769.8376465
1: -574.9697266, 770.4145508, -576.4047852, 772.3870239, -1347.3566895, 1346.8193359
2: -491.2039490, 762.6085815, -492.4450684, 764.5554199, -1255.7592773, 1255.0537109
3: -688.8023071, 923.6856689, -690.5776978, 926.0027466, -1614.8050537, 1614.2634277
4: -651.0784912, 1025.3453369, -652.7169800, 1027.9724121, -1679.0509033, 1678.0622559

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9431277, upper bound: 1541.9401997
time: 0.92 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9401790, upper bound: 1541.9401790
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -763.6300659, 951.8750000, -784.7306519, 980.4318237, -1744.0618896, 1736.6057129
1: -555.7418823, 745.5239258, -574.2277832, 769.3301392, -1325.0720215, 1319.7514648
2: -474.9835510, 737.6679077, -490.5498962, 761.5339966, -1236.5175781, 1228.2176514
3: -666.1837769, 892.9794312, -687.8717651, 922.4364624, -1588.6202393, 1580.8508301
4: -629.3654175, 992.0773926, -650.2122803, 1023.8786011, -1653.2438965, 1642.2896729

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9420778, upper bound: 1541.9337542
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -763.6300659, 951.8750000, -785.9101562, 981.8997803, -1745.5297852, 1737.7851562
1: -555.7418823, 745.5239258, -574.9697266, 770.4145508, -1326.1564941, 1320.4936523
2: -474.9835510, 737.6679077, -491.2039490, 762.6085815, -1237.5921631, 1228.8717041
3: -666.1837769, 892.9794312, -688.8023071, 923.6856689, -1589.8693848, 1581.7813721
4: -629.3654175, 992.0773926, -651.0784912, 1025.3453369, -1654.7104492, 1643.1558838

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9430153, upper bound: 1541.9337720
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9393364, upper bound: 1541.9336964
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -784.7306519, 980.4318237, -763.6300659, 951.8750000, -1736.6057129, 1744.0618896
1: -574.2277832, 769.3301392, -555.7418823, 745.5239258, -1319.7514648, 1325.0720215
2: -490.5498962, 761.5339966, -474.9835510, 737.6679077, -1228.2176514, 1236.5175781
3: -687.8717651, 922.4364624, -666.1837769, 892.9794312, -1580.8508301, 1588.6202393
4: -650.2122803, 1023.8786011, -629.3654175, 992.0773926, -1642.2896729, 1653.2438965

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9337542, upper bound: 1541.9420778
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
time: 0.94 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -785.9101562, 981.8997803, -763.6300659, 951.8750000, -1737.7851562, 1745.5297852
1: -574.9697266, 770.4145508, -555.7418823, 745.5239258, -1320.4936523, 1326.1564941
2: -491.2039490, 762.6085815, -474.9835510, 737.6679077, -1228.8717041, 1237.5921631
3: -688.8023071, 923.6856689, -666.1837769, 892.9794312, -1581.7813721, 1589.8693848
4: -651.0784912, 1025.3453369, -629.3654175, 992.0773926, -1643.1558838, 1654.7104492

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9337720, upper bound: 1541.9430153
time: 0.76 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393364
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -763.6300659, 951.8750000, -760.4378052, 947.9176025, -1711.5476074, 1712.3127441
1: -555.7418823, 745.5239258, -553.5401611, 742.5064087, -1298.2482910, 1299.0640869
2: -474.9835510, 737.6679077, -473.0753174, 734.6762695, -1209.6597900, 1210.7431641
3: -666.1837769, 892.9794312, -663.4779663, 889.4445190, -1555.6281738, 1556.4570312
4: -629.3654175, 992.0773926, -626.8540649, 988.0149536, -1617.3801270, 1618.9313965

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9421287, upper bound: 1541.9337542
time: 0.82 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9315308, upper bound: 1541.9330585
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -763.6300659, 951.8750000, -761.9795532, 949.7575684, -1713.3876953, 1713.8544922
1: -555.7418823, 745.5239258, -554.5682983, 743.8664551, -1299.6083984, 1300.0922852
2: -474.9835510, 737.6679077, -473.9660950, 736.0338135, -1211.0173340, 1211.6340332
3: -666.1837769, 892.9794312, -664.7064819, 891.0440063, -1557.2274170, 1557.6853027
4: -629.3654175, 992.0773926, -628.0175781, 989.8700562, -1619.2352295, 1620.0949707

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9337773, upper bound: 1541.9429374
time: 0.71 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.76 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9408054, upper bound: 1541.9341846
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9431277, upper bound: 1541.9401997
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9401790, upper bound: 1541.9401790
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9420778, upper bound: 1541.9337542
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9430153, upper bound: 1541.9337720
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9393364, upper bound: 1541.9336964
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9337542, upper bound: 1541.9420778
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9337720, upper bound: 1541.9430153
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393364
NS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9421287, upper bound: 1541.9337542
NS_B2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9315308, upper bound: 1541.9330585
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9337773, upper bound: 1541.9429374
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -760.3273315, 949.1828003, -752.4082642, 938.4499512, -1698.7773438, 1701.5910645
1: -556.5480957, 744.8463135, -550.5269775, 736.3262939, -1292.8743896, 1295.3732910
2: -475.3810120, 737.3364258, -470.2539062, 728.9069824, -1204.2879639, 1207.5903320
3: -666.1629028, 893.4214478, -658.6891479, 883.2824097, -1549.4453125, 1552.1105957
4: -630.0595093, 991.2633057, -623.1895142, 979.8875732, -1609.9470215, 1614.4528809

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
time: 1.05 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -742.1736450, 927.1506958, -897.6970215, 1125.8320312, -1868.0056152, 1824.8476562
1: -541.8114624, 726.6697998, -654.9170532, 881.8787842, -1423.6901855, 1381.5867920
2: -463.0528564, 719.5042114, -559.5578003, 873.8446655, -1336.8973389, 1279.0620117
3: -649.0723267, 871.1721191, -786.6316528, 1057.5640869, -1706.6364746, 1657.8037109
4: -613.7495728, 967.5161133, -742.2691650, 1175.2432861, -1788.9925537, 1709.7852783

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
time: 1.03 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -761.2987671, 950.3720703, -752.4082642, 938.4499512, -1699.7486572, 1702.7800293
1: -557.1370239, 745.7227783, -550.5269775, 736.3262939, -1293.4633789, 1296.2497559
2: -475.9076843, 738.1962280, -470.2539062, 728.9069824, -1204.8146973, 1208.4501953
3: -666.9086304, 894.4177246, -658.6891479, 883.2824097, -1550.1909180, 1553.1069336
4: -630.7557373, 992.4368896, -623.1895142, 979.8875732, -1610.6433105, 1615.6263428

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9398852
time: 0.83 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9401790
time: 1.05 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -743.7727051, 929.2545166, -897.6970215, 1125.8320312, -1869.6047363, 1826.9515381
1: -542.8523560, 728.2215576, -654.9170532, 881.8787842, -1424.7308350, 1383.1386719
2: -463.9564209, 721.0523682, -559.5578003, 873.8446655, -1337.8006592, 1280.6101074
3: -650.4120483, 872.9717407, -786.6316528, 1057.5640869, -1707.9760742, 1659.6031494
4: -614.9577026, 969.6348877, -742.2691650, 1175.2432861, -1790.2009277, 1711.9040527

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9401790, upper bound: 1541.9398852
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9401790, upper bound: 1541.9401790
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -737.2621460, 918.3725586, -749.3656616, 934.6717529, -1671.9337158, 1667.7377930
1: -536.5781250, 719.2086792, -548.4672241, 733.4370117, -1270.0151367, 1267.6759033
2: -458.5580444, 711.6689453, -468.4582825, 726.0541382, -1184.6119385, 1180.1269531
3: -642.7177734, 861.7532349, -656.1301880, 879.9159546, -1522.6337891, 1517.8833008
4: -607.5596313, 957.0263672, -620.8171387, 976.0241699, -1583.5837402, 1577.8433838

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
time: 0.99 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
time: 0.95 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -734.1077881, 914.1734009, -894.2971802, 1121.5510254, -1855.6588135, 1808.4704590
1: -532.6968384, 715.2094116, -652.5545654, 878.5811157, -1411.2779541, 1367.7639160
2: -455.5225220, 707.6758423, -557.5065308, 870.5724487, -1326.0949707, 1265.1823730
3: -638.6275635, 856.2561646, -783.6868896, 1053.7010498, -1692.3286133, 1639.9431152
4: -603.4716187, 951.8062134, -739.5656738, 1170.8101807, -1774.2817383, 1691.3715820

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -737.2621460, 918.3725586, -750.3693848, 935.8841553, -1673.1461182, 1668.7415771
1: -536.5781250, 719.2086792, -549.0753784, 734.3292847, -1270.9074707, 1268.2840576
2: -458.5580444, 711.6689453, -469.0012817, 726.9308472, -1185.4887695, 1180.6701660
3: -642.7177734, 861.7532349, -656.8927612, 880.9337769, -1523.6514893, 1518.6458740
4: -607.5596313, 957.0263672, -621.5335083, 977.2179565, -1584.7775879, 1578.5595703

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9393294, upper bound: 1541.9336964
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9393294, upper bound: 1541.9336964
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -734.1077881, 914.1734009, -895.4052734, 1122.9678955, -1857.0756836, 1809.5786133
1: -532.6968384, 715.2094116, -653.3246460, 879.6784058, -1412.3752441, 1368.5340576
2: -455.5225220, 707.6758423, -558.1797485, 871.6618042, -1327.1843262, 1265.8555908
3: -638.6275635, 856.2561646, -784.6664429, 1054.9698486, -1693.5974121, 1640.9223633
4: -603.4716187, 951.8062134, -740.4485474, 1172.3004150, -1775.7719727, 1692.2547607

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9393364, upper bound: 1541.9336964
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9393364, upper bound: 1541.9336964
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -737.2621460, 918.3725586, -1667.7377930, 1671.9337158
1: -548.4672241, 733.4370117, -536.5781250, 719.2086792, -1267.6759033, 1270.0151367
2: -468.4582825, 726.0541382, -458.5580444, 711.6689453, -1180.1269531, 1184.6119385
3: -656.1301880, 879.9159546, -642.7177734, 861.7532349, -1517.8833008, 1522.6337891
4: -620.8171387, 976.0241699, -607.5596313, 957.0263672, -1577.8433838, 1583.5837402

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
time: 0.89 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -734.1077881, 914.1734009, -1808.4705811, 1855.6588135
1: -652.5545654, 878.5811157, -532.6968384, 715.2094116, -1367.7639160, 1411.2779541
2: -557.5065308, 870.5724487, -455.5225220, 707.6758423, -1265.1823730, 1326.0949707
3: -783.6868896, 1053.7010498, -638.6275635, 856.2561646, -1639.9431152, 1692.3286133
4: -739.5656738, 1170.8101807, -603.4716187, 951.8062134, -1691.3717041, 1774.2817383

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
time: 0.82 seconds

## BFS NS instance: NS_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -737.2621460, 918.3725586, -1668.7415771, 1673.1461182
1: -549.0753784, 734.3292847, -536.5781250, 719.2086792, -1268.2840576, 1270.9074707
2: -469.0012817, 726.9308472, -458.5580444, 711.6689453, -1180.6701660, 1185.4887695
3: -656.8927612, 880.9337769, -642.7177734, 861.7532349, -1518.6458740, 1523.6514893
4: -621.5335083, 977.2179565, -607.5596313, 957.0263672, -1578.5595703, 1584.7775879

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393294
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393294
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -734.1077881, 914.1734009, -1809.5786133, 1857.0756836
1: -653.3246460, 879.6784058, -532.6968384, 715.2094116, -1368.5340576, 1412.3752441
2: -558.1797485, 871.6618042, -455.5225220, 707.6758423, -1265.8555908, 1327.1843262
3: -784.6664429, 1054.9698486, -638.6275635, 856.2561646, -1640.9223633, 1693.5974121
4: -740.4485474, 1172.3004150, -603.4716187, 951.8062134, -1692.2547607, 1775.7719727

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393364
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393364
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -737.2621460, 918.3725586, -718.3277588, 894.0231934, -1631.2852783, 1636.7000732
1: -536.5781250, 719.2086792, -522.8232422, 700.1860352, -1236.7641602, 1242.0319824
2: -458.5580444, 711.6689453, -446.7558289, 692.8474731, -1151.4051514, 1158.4245605
3: -642.7177734, 861.7532349, -625.7783813, 839.2131348, -1481.9309082, 1487.5316162
4: -607.5596313, 957.0263672, -591.8724976, 931.6036377, -1539.1630859, 1548.8985596

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9315308, upper bound: 1541.9330585
time: 0.68 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9315308, upper bound: 1541.9330585
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -735.5803833, 916.2291870, -1637.8203125, 1633.7590332
1: -525.0841064, 703.3433228, -535.3843994, 717.5314331, -1242.6154785, 1238.7275391
2: -448.7170715, 695.9824829, -457.5234680, 710.0163574, -1158.7331543, 1153.5058594
3: -628.5903320, 842.9027710, -641.2233276, 859.7930298, -1488.3833008, 1484.1260986
4: -594.4638672, 935.8617554, -606.1907959, 954.7958374, -1549.2597656, 1542.0524902

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
time: 0.74 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -732.6508789, 912.3205566, -1792.9752197, 1835.8465576
1: -641.1992188, 863.6813354, -531.6658325, 713.7680664, -1354.9672852, 1395.3470459
2: -548.1012573, 855.3493042, -454.6305542, 706.2503052, -1254.3514404, 1309.9798584
3: -770.0324097, 1035.2294922, -637.3362427, 854.5688477, -1624.6013184, 1672.5656738
4: -726.8616943, 1150.2670898, -602.2912598, 949.8806763, -1676.7420654, 1752.5583496

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
time: 0.74 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.88 seconds
NS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
NS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
NS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
NS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9375403, upper bound: 1541.9341200
NS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9398852
NS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9401790
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9401790, upper bound: 1541.9398852
NS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9401790, upper bound: 1541.9401790
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9340161, upper bound: 1541.9334687
NS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9393294, upper bound: 1541.9336964
NS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9393294, upper bound: 1541.9336964
NS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9393364, upper bound: 1541.9336964
NS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9393364, upper bound: 1541.9336964
NS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
NS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9334687, upper bound: 1541.9340161
NS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393294
NS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393294
NS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393364
NS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9336964, upper bound: 1541.9393364
NS_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9315308, upper bound: 1541.9330585
NS_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9315308, upper bound: 1541.9330585
NS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
NS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
NS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221
NS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.88
Output dim: 0, lower bound: -1541.9335220, upper bound: 1541.9335221

## BFS NS instance: NS_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -752.4082642, 938.4499512, -1687.8153076, 1687.0798340
1: -548.4672241, 733.4370117, -550.5269775, 736.3262939, -1284.7934570, 1283.9639893
2: -468.4582825, 726.0541382, -470.2539062, 728.9069824, -1197.3652344, 1196.3079834
3: -656.1301880, 879.9159546, -658.6891479, 883.2824097, -1539.4125977, 1538.6051025
4: -620.8171387, 976.0241699, -623.1895142, 979.8875732, -1600.7047119, 1599.2136230

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317884
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9377671, upper bound: 1541.9317892
time: 0.96 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -891.8333740, 1118.7094727, -752.4082642, 938.4499512, -1830.2833252, 1871.1176758
1: -650.8681641, 876.3784790, -550.5269775, 736.3262939, -1387.1944580, 1426.9053955
2: -556.0549316, 868.4358521, -470.2539062, 728.9069824, -1284.9617920, 1338.6896973
3: -781.7049561, 1051.1008301, -658.6891479, 883.2824097, -1664.9873047, 1709.7900391
4: -737.6659546, 1167.9645996, -623.1895142, 979.8875732, -1717.5534668, 1791.1540527

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B1_A2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9388113, upper bound: 1541.9336682
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9377671, upper bound: 1541.9317892
time: 0.91 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -897.6970215, 1125.8320312, -1875.1976318, 1832.3687744
1: -548.4672241, 733.4370117, -654.9170532, 881.8787842, -1430.3459473, 1388.3538818
2: -468.4582825, 726.0541382, -559.5578003, 873.8446655, -1342.3026123, 1285.6119385
3: -656.1301880, 879.9159546, -786.6316528, 1057.5640869, -1713.6942139, 1666.5476074
4: -620.8171387, 976.0241699, -742.2691650, 1175.2432861, -1796.0604248, 1718.2933350

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9352659, upper bound: 1541.9317388
time: 0.86 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9323404, upper bound: 1541.9315736
time: 0.71 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -897.6970215, 1125.8320312, -2020.1291504, 2019.2480469
1: -652.5545654, 878.5811157, -654.9170532, 881.8787842, -1534.4333496, 1533.4981689
2: -557.5065308, 870.5724487, -559.5578003, 873.8446655, -1431.3511963, 1430.1302490
3: -783.6868896, 1053.7010498, -786.6316528, 1057.5640869, -1841.2509766, 1840.3327637
4: -739.5656738, 1170.8101807, -742.2691650, 1175.2432861, -1914.8085938, 1913.0793457

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9339626, upper bound: 1541.9339626
time: 0.95 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9339626, upper bound: 1541.9341200
time: 0.97 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -752.4082642, 938.4499512, -1688.8192139, 1688.2922363
1: -549.0753784, 734.3292847, -550.5269775, 736.3262939, -1285.4016113, 1284.8562012
2: -469.0012817, 726.9308472, -470.2539062, 728.9069824, -1197.9082031, 1197.1848145
3: -656.8927612, 880.9337769, -658.6891479, 883.2824097, -1540.1751709, 1539.6229248
4: -621.5335083, 977.2179565, -623.1895142, 979.8875732, -1601.4211426, 1600.4074707

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9431277, upper bound: 1541.9399185
time: 0.95 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9423581, upper bound: 1541.9399185
time: 1.01 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -892.8350830, 1119.9968262, -752.4082642, 938.4499512, -1831.2850342, 1872.4050293
1: -651.5625000, 877.3747559, -550.5269775, 736.3262939, -1387.8887939, 1427.9013672
2: -556.6630249, 869.4265137, -470.2539062, 728.9069824, -1285.5700684, 1339.6800537
3: -782.5938721, 1052.2486572, -658.6891479, 883.2824097, -1665.8762207, 1710.9377441
4: -738.4631348, 1169.3232422, -623.1895142, 979.8875732, -1718.3505859, 1792.5126953

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9416095, upper bound: 1541.9401624
time: 0.76 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9423581, upper bound: 1541.9401624
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -897.6970215, 1125.8320312, -1876.2012939, 1833.5811768
1: -549.0753784, 734.3292847, -654.9170532, 881.8787842, -1430.9541016, 1389.2462158
2: -469.0012817, 726.9308472, -559.5578003, 873.8446655, -1342.8458252, 1286.4886475
3: -656.8927612, 880.9337769, -786.6316528, 1057.5640869, -1714.4567871, 1667.5654297
4: -621.5335083, 977.2179565, -742.2691650, 1175.2432861, -1796.7766113, 1719.4870605

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9398852
time: 0.98 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9398852
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -897.6970215, 1125.8320312, -2021.2373047, 2020.6649170
1: -653.3246460, 879.6784058, -654.9170532, 881.8787842, -1535.2033691, 1534.5954590
2: -558.1797485, 871.6618042, -559.5578003, 873.8446655, -1432.0241699, 1431.2196045
3: -784.6664429, 1054.9698486, -786.6316528, 1057.5640869, -1842.2304688, 1841.6015625
4: -740.4485474, 1172.3004150, -742.2691650, 1175.2432861, -1915.6917725, 1914.5695801

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9341200, upper bound: 1541.9375403
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9341200, upper bound: 1541.9401790
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -749.3656616, 934.6717529, -1656.2628174, 1647.5441895
1: -525.0841064, 703.3433228, -548.4672241, 733.4370117, -1258.5211182, 1251.8104248
2: -448.7170715, 695.9824829, -468.4582825, 726.0541382, -1174.7708740, 1164.4406738
3: -628.5903320, 842.9027710, -656.1301880, 879.9159546, -1508.5063477, 1499.0329590
4: -594.4638672, 935.8617554, -620.8171387, 976.0241699, -1570.4880371, 1556.6789551

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9364714, upper bound: 1541.9315150
time: 0.89 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9352643, upper bound: 1541.9314543
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -749.3656616, 934.6717529, -1815.3264160, 1852.5611572
1: -641.1992188, 863.6813354, -548.4672241, 733.4370117, -1374.6359863, 1412.1485596
2: -548.1012573, 855.3493042, -468.4582825, 726.0541382, -1274.1551514, 1323.8074951
3: -770.0324097, 1035.2294922, -656.1301880, 879.9159546, -1649.9483643, 1691.3596191
4: -726.8616943, 1150.2670898, -620.8171387, 976.0241699, -1702.8858643, 1771.0842285

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9316173
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9337542
time: 0.80 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -894.2971802, 1121.5510254, -1843.1422119, 1792.4758301
1: -525.0841064, 703.3433228, -652.5545654, 878.5811157, -1403.6652832, 1355.8978271
2: -448.7170715, 695.9824829, -557.5065308, 870.5724487, -1319.2894287, 1253.4890137
3: -628.5903320, 842.9027710, -783.6868896, 1053.7010498, -1682.2913818, 1626.5895996
4: -594.4638672, 935.8617554, -739.5656738, 1170.8101807, -1765.2739258, 1675.4272461

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334100, upper bound: 1541.9314342
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9315566, upper bound: 1541.9313221
time: 0.83 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -894.2971802, 1121.5510254, -2002.2056885, 1997.4929199
1: -641.1992188, 863.6813354, -652.5545654, 878.5811157, -1519.7802734, 1516.2358398
2: -548.1012573, 855.3493042, -557.5065308, 870.5724487, -1418.6737061, 1412.8558350
3: -770.0324097, 1035.2294922, -783.6868896, 1053.7010498, -1823.7333984, 1818.9163818
4: -726.8616943, 1150.2670898, -739.5656738, 1170.8101807, -1897.6717529, 1889.8325195

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9315380
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9334687
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -750.3693848, 935.8841553, -1657.4752197, 1648.5480957
1: -525.0841064, 703.3433228, -549.0753784, 734.3292847, -1259.4133301, 1252.4185791
2: -448.7170715, 695.9824829, -469.0012817, 726.9308472, -1175.6477051, 1164.9837646
3: -628.5903320, 842.9027710, -656.8927612, 880.9337769, -1509.5240479, 1499.7955322
4: -594.4638672, 935.8617554, -621.5335083, 977.2179565, -1571.6818848, 1557.3952637

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9376222, upper bound: 1541.9315322
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9377436, upper bound: 1541.9315333
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -750.3693848, 935.8841553, -1816.5388184, 1853.5649414
1: -641.1992188, 863.6813354, -549.0753784, 734.3292847, -1375.5284424, 1412.7567139
2: -548.1012573, 855.3493042, -469.0012817, 726.9308472, -1275.0319824, 1324.3505859
3: -770.0324097, 1035.2294922, -656.8927612, 880.9337769, -1650.9661865, 1692.1223145
4: -726.8616943, 1150.2670898, -621.5335083, 977.2179565, -1704.0795898, 1771.8005371

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9316349
time: 0.98 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9337720
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -895.4052734, 1122.9678955, -1844.5590820, 1793.5839844
1: -525.0841064, 703.3433228, -653.3246460, 879.6784058, -1404.7624512, 1356.6677246
2: -448.7170715, 695.9824829, -558.1797485, 871.6618042, -1320.3785400, 1254.1622314
3: -628.5903320, 842.9027710, -784.6664429, 1054.9698486, -1683.5601807, 1627.5690918
4: -594.4638672, 935.8617554, -740.4485474, 1172.3004150, -1766.7641602, 1676.3103027

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9352619, upper bound: 1541.9314902
time: 1.03 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9323279, upper bound: 1541.9313269
time: 0.80 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -895.4052734, 1122.9678955, -2003.6225586, 1998.6009521
1: -641.1992188, 863.6813354, -653.3246460, 879.6784058, -1520.8776855, 1517.0059814
2: -548.1012573, 855.3493042, -558.1797485, 871.6618042, -1419.7628174, 1413.5290527
3: -770.0324097, 1035.2294922, -784.6664429, 1054.9698486, -1825.0021973, 1819.8957520
4: -726.8616943, 1150.2670898, -740.4485474, 1172.3004150, -1899.1619873, 1890.7155762

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9315976
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9336964
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -721.5911865, 898.1787109, -1647.5441895, 1656.2628174
1: -548.4672241, 733.4370117, -525.0841064, 703.3433228, -1251.8104248, 1258.5211182
2: -468.4582825, 726.0541382, -448.7170715, 695.9824829, -1164.4406738, 1174.7708740
3: -656.1301880, 879.9159546, -628.5903320, 842.9027710, -1499.0329590, 1508.5063477
4: -620.8171387, 976.0241699, -594.4638672, 935.8617554, -1556.6789551, 1570.4880371

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315152, upper bound: 1541.9364714
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9314543, upper bound: 1541.9352643
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -880.6546631, 1103.1956787, -1852.5611572, 1815.3264160
1: -548.4672241, 733.4370117, -641.1992188, 863.6813354, -1412.1485596, 1374.6361084
2: -468.4582825, 726.0541382, -548.1012573, 855.3493042, -1323.8074951, 1274.1551514
3: -656.1301880, 879.9159546, -770.0324097, 1035.2294922, -1691.3596191, 1649.9483643
4: -620.8171387, 976.0241699, -726.8616943, 1150.2670898, -1771.0842285, 1702.8858643

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9316174, upper bound: 1541.9366471
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9316174, upper bound: 1541.9420778
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -721.5911865, 898.1787109, -1792.4758301, 1843.1422119
1: -652.5545654, 878.5811157, -525.0841064, 703.3433228, -1355.8978271, 1403.6652832
2: -557.5065308, 870.5724487, -448.7170715, 695.9824829, -1253.4890137, 1319.2894287
3: -783.6868896, 1053.7010498, -628.5903320, 842.9027710, -1626.5895996, 1682.2913818
4: -739.5656738, 1170.8101807, -594.4638672, 935.8617554, -1675.4272461, 1765.2740479

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9314342, upper bound: 1541.9334100
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9313221, upper bound: 1541.9315566
time: 0.74 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -880.6546631, 1103.1956787, -1997.4929199, 2002.2056885
1: -652.5545654, 878.5811157, -641.1992188, 863.6813354, -1516.2358398, 1519.7802734
2: -557.5065308, 870.5724487, -548.1012573, 855.3493042, -1412.8558350, 1418.6737061
3: -783.6868896, 1053.7010498, -770.0324097, 1035.2294922, -1818.9163818, 1823.7333984
4: -739.5656738, 1170.8101807, -726.8616943, 1150.2670898, -1889.8325195, 1897.6717529

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315380, upper bound: 1541.9338357
time: 0.93 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315380, upper bound: 1541.9340161
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -721.5911865, 898.1787109, -1648.5480957, 1657.4752197
1: -549.0753784, 734.3292847, -525.0841064, 703.3433228, -1252.4185791, 1259.4133301
2: -469.0012817, 726.9308472, -448.7170715, 695.9824829, -1164.9837646, 1175.6477051
3: -656.8927612, 880.9337769, -628.5903320, 842.9027710, -1499.7955322, 1509.5240479
4: -621.5335083, 977.2179565, -594.4638672, 935.8617554, -1557.3952637, 1571.6818848

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A1_B1_A1

### Relational analysis result of NS_B2_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315323, upper bound: 1541.9376222
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A2_A1_B1_A2

### Relational analysis result of NS_B2_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315334, upper bound: 1541.9377436
time: 1.14 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -880.6546631, 1103.1956787, -1853.5649414, 1816.5388184
1: -549.0753784, 734.3292847, -641.1992188, 863.6813354, -1412.7567139, 1375.5283203
2: -469.0012817, 726.9308472, -548.1012573, 855.3493042, -1324.3505859, 1275.0319824
3: -656.8927612, 880.9337769, -770.0324097, 1035.2294922, -1692.1223145, 1650.9661865
4: -621.5335083, 977.2179565, -726.8616943, 1150.2670898, -1771.8005371, 1704.0795898

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9316349, upper bound: 1541.9379462
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9316349, upper bound: 1541.9430153
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -721.5911865, 898.1787109, -1793.5839844, 1844.5590820
1: -653.3246460, 879.6784058, -525.0841064, 703.3433228, -1356.6677246, 1404.7624512
2: -558.1797485, 871.6618042, -448.7170715, 695.9824829, -1254.1622314, 1320.3785400
3: -784.6664429, 1054.9698486, -628.5903320, 842.9027710, -1627.5692139, 1683.5601807
4: -740.4485474, 1172.3004150, -594.4638672, 935.8617554, -1676.3103027, 1766.7642822

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9314902, upper bound: 1541.9352619
time: 1.00 seconds

## Relational analysis of NS_B2_A1_A2_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9313268, upper bound: 1541.9323279
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -880.6546631, 1103.1956787, -1998.6009521, 2003.6225586
1: -653.3246460, 879.6784058, -641.1992188, 863.6813354, -1517.0059814, 1520.8776855
2: -558.1797485, 871.6618042, -548.1012573, 855.3493042, -1413.5290527, 1419.7628174
3: -784.6664429, 1054.9698486, -770.0324097, 1035.2294922, -1819.8957520, 1825.0021973
4: -740.4485474, 1172.3004150, -726.8616943, 1150.2670898, -1890.7155762, 1899.1619873

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A2_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315976, upper bound: 1541.9356193
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315976, upper bound: 1541.9393364
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -719.8602295, 895.9888916, -1617.5799561, 1618.0388184
1: -525.0841064, 703.3433228, -523.8558350, 701.6317139, -1226.7156982, 1227.1989746
2: -448.7170715, 695.9824829, -447.6536560, 694.2944946, -1143.0113525, 1143.6359863
3: -628.5903320, 842.9027710, -627.0578003, 840.9013062, -1469.4916992, 1469.9605713
4: -594.4638672, 935.8617554, -593.0591431, 933.5846558, -1528.0485840, 1528.9208984

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9317161, upper bound: 1541.9423354
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9317161, upper bound: 1541.9429374
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -721.5911865, 898.1787109, -878.4260254, 1100.3701172, -1821.9611816, 1776.6047363
1: -525.0841064, 703.3433228, -639.6423950, 861.5164795, -1386.6005859, 1342.9855957
2: -448.7170715, 695.9824829, -546.7550659, 853.1940308, -1301.9108887, 1242.7374268
3: -628.5903320, 842.9027710, -768.1010742, 1032.6743164, -1661.2646484, 1611.0039062
4: -594.4638672, 935.8617554, -725.0787354, 1147.3596191, -1741.8233643, 1660.9404297

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9329866, upper bound: 1541.9419781
time: 0.92 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9316091, upper bound: 1541.9404766
time: 0.97 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -719.8602295, 895.9888916, -1776.6435547, 1823.0557861
1: -641.1992188, 863.6813354, -523.8558350, 701.6317139, -1342.8306885, 1387.5371094
2: -548.1012573, 855.3493042, -447.6536560, 694.2944946, -1242.3957520, 1303.0028076
3: -770.0324097, 1035.2294922, -627.0578003, 840.9013062, -1610.9337158, 1662.2873535
4: -726.8616943, 1150.2670898, -593.0591431, 933.5846558, -1660.4462891, 1743.3261719

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9315308
time: 1.30 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9335221
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -880.6546631, 1103.1956787, -878.4260254, 1100.3701172, -1981.0247803, 1981.6217041
1: -641.1992188, 863.6813354, -639.6423950, 861.5164795, -1502.7156982, 1503.3237305
2: -548.1012573, 855.3493042, -546.7550659, 853.1940308, -1401.2951660, 1402.1042480
3: -770.0324097, 1035.2294922, -768.1010742, 1032.6743164, -1802.7067871, 1803.3305664
4: -726.8616943, 1150.2670898, -725.0787354, 1147.3596191, -1874.2211914, 1875.3458252

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9315307
time: 1.08 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9335221
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.90 seconds
NS_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317884
NS_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9377671, upper bound: 1541.9317892
NS_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9388113, upper bound: 1541.9336682
NS_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9377671, upper bound: 1541.9317892
NS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9352659, upper bound: 1541.9317388
NS_B1_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9323404, upper bound: 1541.9315736
NS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9339626, upper bound: 1541.9339626
NS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9339626, upper bound: 1541.9341200
NS_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9431277, upper bound: 1541.9399185
NS_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9423581, upper bound: 1541.9399185
NS_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9416095, upper bound: 1541.9401624
NS_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9423581, upper bound: 1541.9401624
NS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9398852
NS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9398852, upper bound: 1541.9398852
NS_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9341200, upper bound: 1541.9375403
NS_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9341200, upper bound: 1541.9401790
NS_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9364714, upper bound: 1541.9315150
NS_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9352643, upper bound: 1541.9314543
NS_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9316173
NS_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9337542
NS_B1_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9334100, upper bound: 1541.9314342
NS_B1_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315566, upper bound: 1541.9313221
NS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9315380
NS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9334687
NS_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9376222, upper bound: 1541.9315322
NS_B1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9377436, upper bound: 1541.9315333
NS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9316349
NS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9366471, upper bound: 1541.9337720
NS_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9352619, upper bound: 1541.9314902
NS_B1_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9323279, upper bound: 1541.9313269
NS_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9315976
NS_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9338357, upper bound: 1541.9336964
NS_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315152, upper bound: 1541.9364714
NS_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314543, upper bound: 1541.9352643
NS_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9316174, upper bound: 1541.9366471
NS_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9316174, upper bound: 1541.9420778
NS_B2_A1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314342, upper bound: 1541.9334100
NS_B2_A1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9313221, upper bound: 1541.9315566
NS_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315380, upper bound: 1541.9338357
NS_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315380, upper bound: 1541.9340161
NS_B2_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315323, upper bound: 1541.9376222
NS_B2_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315334, upper bound: 1541.9377436
NS_B2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9316349, upper bound: 1541.9379462
NS_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9316349, upper bound: 1541.9430153
NS_B2_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314902, upper bound: 1541.9352619
NS_B2_A1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9313268, upper bound: 1541.9323279
NS_B2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315976, upper bound: 1541.9356193
NS_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9315976, upper bound: 1541.9393364
NS_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9317161, upper bound: 1541.9423354
NS_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9317161, upper bound: 1541.9429374
NS_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9329866, upper bound: 1541.9419781
NS_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9316091, upper bound: 1541.9404766
NS_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9315308
NS_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9335221
NS_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9315307
NS_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -1541.9314111, upper bound: 1541.9335221

## BFS NS instance: NS_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -720.3057251, 899.4554443, -701.2141113, 878.2443848, -1598.5500488, 1600.6695557
1: -527.4898682, 705.9353027, -517.4451904, 691.7836914, -1219.2734375, 1223.3804932
2: -450.5196533, 699.0070190, -441.7453613, 685.2877808, -1135.8073730, 1140.7521973
3: -631.0921631, 847.3621826, -618.5140991, 831.5227051, -1462.6148682, 1465.8760986
4: -597.2335815, 939.7742310, -585.9918213, 921.6644287, -1518.8979492, 1525.7658691

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355307
time: 1.02 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355307
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -738.4709473, 920.2429199, -736.9458008, 917.7936401, -1656.2646484, 1657.1887207
1: -540.0728760, 722.0300293, -538.5001831, 719.9794922, -1260.0523682, 1260.5302734
2: -461.3500671, 714.6257935, -460.0958557, 712.5017090, -1173.8515625, 1174.7216797
3: -645.9631348, 866.1131592, -644.1045532, 863.4508057, -1509.4139404, 1510.2176514
4: -611.2856445, 960.6207886, -609.5385742, 957.7708130, -1569.0562744, 1570.1594238

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355309
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355307
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -865.7438354, 1091.2446289, -722.9949951, 902.8814697, -1768.6252441, 1814.2396240
1: -638.0795288, 857.7741699, -529.3288574, 708.5452881, -1346.6247559, 1387.1029053
2: -544.6342163, 850.7475586, -452.1165466, 701.5881958, -1246.2224121, 1302.8641357
3: -764.8753052, 1031.1295166, -633.3886719, 850.4039917, -1615.2792969, 1664.5181885
4: -723.1636353, 1144.4539795, -599.3504639, 943.2697754, -1666.4333496, 1743.8044434

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317885
time: 0.89 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317892
time: 0.92 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -874.3792114, 1095.4095459, -741.1099243, 923.5119629, -1797.8911133, 1836.5195312
1: -637.1982422, 857.7413330, -541.8330078, 724.5128174, -1361.7110596, 1399.5743408
2: -544.4678345, 849.8716431, -462.8899231, 717.0772095, -1261.5450439, 1312.7614746
3: -765.0127563, 1028.6613770, -648.1621094, 868.9938354, -1634.0065918, 1676.8233643
4: -722.1173706, 1142.8398438, -613.3171387, 963.9432373, -1686.0605469, 1756.1569824

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317885
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317893
time: 0.91 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -720.3057251, 899.4554443, -870.6052246, 1097.1925049, -1817.4981689, 1770.0606689
1: -527.4898682, 705.9353027, -641.4211426, 862.3569946, -1389.8469238, 1347.3564453
2: -450.5196533, 699.0070190, -547.5345459, 855.2782593, -1305.7978516, 1246.5411377
3: -631.0921631, 847.3621826, -768.9739990, 1036.4984131, -1667.5903320, 1616.3361816
4: -597.2335815, 939.7742310, -726.9722900, 1150.5625000, -1747.7961426, 1666.7462158

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9324742, upper bound: 1541.9352828
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9324742, upper bound: 1541.9352828
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -894.2971802, 1121.5510254, -2015.8481445, 2015.8481445
1: -652.5545654, 878.5811157, -652.5545654, 878.5811157, -1531.1357422, 1531.1357422
2: -557.5065308, 870.5724487, -557.5065308, 870.5724487, -1428.0789795, 1428.0789795
3: -783.6868896, 1053.7010498, -783.6868896, 1053.7010498, -1837.3879395, 1837.3879395
4: -739.5656738, 1170.8101807, -739.5656738, 1170.8101807, -1910.3756104, 1910.3756104

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9301761, upper bound: 1541.9247283
time: 1.16 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9244989, upper bound: 1541.9244661
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -895.4052734, 1122.9678955, -2017.2651367, 2016.9562988
1: -652.5545654, 878.5811157, -653.3246460, 879.6784058, -1532.2329102, 1531.9057617
2: -557.5065308, 870.5724487, -558.1797485, 871.6618042, -1429.1683350, 1428.7521973
3: -783.6868896, 1053.7010498, -784.6664429, 1054.9698486, -1838.6567383, 1838.3674316
4: -739.5656738, 1170.8101807, -740.4485474, 1172.3004150, -1911.8658447, 1911.2587891

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9301761, upper bound: 1541.9247413
time: 0.83 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9244989, upper bound: 1541.9245197
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -738.4054565, 923.0206909, -706.1461792, 882.2407837, -1620.6461182, 1629.1665039
1: -545.1910400, 727.1314087, -517.1881714, 692.2742310, -1237.4653320, 1244.3195801
2: -464.8539429, 719.2363281, -441.8111877, 685.4320679, -1150.2860107, 1161.0474854
3: -652.2098999, 874.2393188, -619.0485840, 830.6516724, -1482.8614502, 1493.2878418
4: -616.8860474, 965.7790527, -585.6623535, 921.7138062, -1538.5998535, 1551.4414062

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9434022
time: 1.22 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9434022
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -733.8441772, 915.1110840, -741.3265991, 924.5228882, -1658.3670654, 1656.4377441
1: -537.1129761, 718.0685425, -542.5028687, 725.4182739, -1262.5311279, 1260.5714111
2: -458.7411194, 710.8515015, -463.3732605, 718.1257935, -1176.8666992, 1174.2246094
3: -642.4709473, 861.5037842, -649.0119019, 870.2498169, -1512.7207031, 1510.5156250
4: -607.9426880, 955.5560303, -614.0726929, 965.3659058, -1573.3085938, 1569.6286621

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9436910
time: 0.77 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9436910
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -846.8486938, 1064.2666016, -740.4943848, 925.6234741, -1772.4721680, 1804.7606201
1: -619.6625366, 834.2979736, -546.6649780, 729.1660156, -1348.8284912, 1380.9628906
2: -529.2944946, 826.8439331, -466.1278381, 721.2433472, -1250.5375977, 1292.9716797
3: -744.0665894, 1001.3145142, -654.0315552, 876.6279907, -1620.6945801, 1655.3459473
4: -702.4167480, 1112.2340088, -618.5678711, 968.4912109, -1670.9078369, 1730.8015137

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9416095, upper bound: 1541.9401624
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9416095, upper bound: 1541.9401624
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -880.7676392, 1104.3321533, -735.9094238, 917.7145996, -1798.4821777, 1840.2410889
1: -642.5864258, 865.0694580, -538.5885620, 720.0979614, -1362.6843262, 1403.6579590
2: -549.0008545, 857.1881104, -460.0127258, 712.8598022, -1261.8605957, 1317.2008057
3: -771.6562500, 1037.5137939, -644.2976074, 863.8927612, -1635.5490723, 1681.8112793
4: -728.2492065, 1152.7904053, -609.6248779, 958.2683105, -1686.5175781, 1762.4152832

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9343570, upper bound: 1541.9353113
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9392565, upper bound: 1541.9374877
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -738.4054565, 923.0206909, -851.6403198, 1069.9779053, -1808.3833008, 1774.6608887
1: -545.1910400, 727.1314087, -622.9891968, 838.7200317, -1383.9111328, 1350.1206055
2: -464.8539429, 719.2363281, -532.1625366, 831.1641846, -1296.0180664, 1251.3988037
3: -652.2098999, 874.2393188, -748.0437622, 1006.5432129, -1658.7530518, 1622.2829590
4: -616.8860474, 965.7790527, -706.1859741, 1118.0117188, -1734.8975830, 1671.9649658

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9401624, upper bound: 1541.9416095
time: 1.04 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9401624, upper bound: 1541.9416095
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -733.8441772, 915.1110840, -885.6356201, 1110.1762695, -1844.0203857, 1800.7467041
1: -537.1129761, 718.0685425, -645.9489136, 869.5803223, -1406.6932373, 1364.0174561
2: -458.7411194, 710.8515015, -551.9008789, 861.6140137, -1320.3549805, 1262.7524414
3: -642.4709473, 861.5037842, -775.7010498, 1042.8387451, -1685.3096924, 1637.2048340
4: -607.9426880, 955.5560303, -732.0616455, 1158.7222900, -1766.6647949, 1687.6176758

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9353112, upper bound: 1541.9343570
time: 0.91 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9374877, upper bound: 1541.9392565
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -894.2971802, 1121.5510254, -2016.9562988, 2017.2651367
1: -653.3246460, 879.6784058, -652.5545654, 878.5811157, -1531.9057617, 1532.2329102
2: -558.1797485, 871.6618042, -557.5065308, 870.5724487, -1428.7521973, 1429.1683350
3: -784.6664429, 1054.9698486, -783.6868896, 1053.7010498, -1838.3674316, 1838.6567383
4: -740.4485474, 1172.3004150, -739.5656738, 1170.8101807, -1911.2587891, 1911.8658447

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9333753, upper bound: 1541.9302701
time: 0.77 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9341200, upper bound: 1541.9375403
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -895.4052734, 1122.9678955, -2018.3731689, 2018.3731689
1: -653.3246460, 879.6784058, -653.3246460, 879.6784058, -1533.0030518, 1533.0030518
2: -558.1797485, 871.6618042, -558.1797485, 871.6618042, -1429.8414307, 1429.8414307
3: -784.6664429, 1054.9698486, -784.6664429, 1054.9698486, -1839.6361084, 1839.6361084
4: -740.4485474, 1172.3004150, -740.4485474, 1172.3004150, -1912.7490234, 1912.7490234

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9315223, upper bound: 1541.9263113
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9245434, upper bound: 1541.9260528
time: 1.07 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -699.8526611, 870.4273682, -697.6857910, 873.7763672, -1573.6290283, 1568.1130371
1: -508.8080750, 681.4273071, -515.0140991, 688.3702393, -1197.1781006, 1196.4411621
2: -434.8721313, 674.2592163, -439.6369324, 681.8969727, -1116.7690430, 1113.8961182
3: -608.8891602, 816.8630981, -615.4884033, 827.5158691, -1436.4050293, 1432.3510742
4: -576.1090088, 906.6450806, -583.1974487, 917.0775757, -1493.1865234, 1489.8425293

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9446024
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9450699
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -703.1428223, 873.5878296, -734.2792969, 914.4768677, -1617.6196289, 1607.8671875
1: -511.0803223, 684.1479492, -536.7048950, 717.4542847, -1228.5345459, 1220.8527832
2: -436.8538208, 676.7639160, -458.5287476, 710.0073242, -1146.8610840, 1135.2924805
3: -611.4164429, 819.8616333, -641.8637085, 860.5128174, -1471.9290771, 1461.7250977
4: -578.5692749, 909.9353638, -607.4680786, 954.3909912, -1532.9602051, 1517.4029541

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9356442, upper bound: 1541.9409829
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9356442, upper bound: 1541.9434154
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -877.4686279, 1099.2493896, -749.3656616, 934.6717529, -1812.1402588, 1848.6148682
1: -639.0079956, 860.6503906, -548.4672241, 733.4370117, -1372.4450684, 1409.1174316
2: -546.1976318, 852.3526001, -468.4582825, 726.0541382, -1272.2515869, 1320.8104248
3: -767.3111572, 1031.6871338, -656.1301880, 879.9159546, -1647.2270508, 1687.8171387
4: -724.3611450, 1146.1955566, -620.8171387, 976.0241699, -1700.3852539, 1767.0126953

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9290016
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9298419
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -878.4260254, 1100.3701172, -749.3656616, 934.6717529, -1813.0977783, 1849.7357178
1: -639.6423950, 861.5164795, -548.4672241, 733.4370117, -1373.0793457, 1409.9836426
2: -546.7550659, 853.1940308, -468.4582825, 726.0541382, -1272.8089600, 1321.6522217
3: -768.1010742, 1032.6743164, -656.1301880, 879.9159546, -1648.0170898, 1688.8044434
4: -725.0787354, 1147.3596191, -620.8171387, 976.0241699, -1701.1029053, 1768.1767578

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9295638
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9324271
time: 1.06 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -877.4686279, 1099.2493896, -894.2971802, 1121.5510254, -1999.0196533, 1993.5465088
1: -639.0079956, 860.6503906, -652.5545654, 878.5811157, -1517.5891113, 1513.2049561
2: -546.1976318, 852.3526001, -557.5065308, 870.5724487, -1416.7700195, 1409.8590088
3: -767.3111572, 1031.6871338, -783.6868896, 1053.7010498, -1821.0122070, 1815.3740234
4: -724.3611450, 1146.1955566, -739.5656738, 1170.8101807, -1895.1713867, 1885.7609863

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9241006, upper bound: 1541.9243067
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9241078, upper bound: 1541.9233540
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -878.4260254, 1100.3701172, -894.2971802, 1121.5510254, -1999.9770508, 1994.6672363
1: -639.6423950, 861.5164795, -652.5545654, 878.5811157, -1518.2235107, 1514.0710449
2: -546.7550659, 853.1940308, -557.5065308, 870.5724487, -1417.3275146, 1410.7005615
3: -768.1010742, 1032.6743164, -783.6868896, 1053.7010498, -1821.8021240, 1816.3612061
4: -725.0787354, 1147.3596191, -739.5656738, 1170.8101807, -1895.8889160, 1886.9250488

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334097, upper bound: 1541.9314342
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9315564, upper bound: 1541.9313221
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -699.8526611, 870.4273682, -699.0020142, 875.4768677, -1575.3294678, 1569.4291992
1: -508.8080750, 681.4273071, -515.8696289, 689.6176147, -1198.4256592, 1197.2967529
2: -434.8721313, 674.2592163, -440.3870544, 683.1488037, -1118.0209961, 1114.6462402
3: -608.8891602, 816.8630981, -616.5712280, 828.9740601, -1437.8632812, 1433.4340820
4: -576.1090088, 906.6450806, -584.1976929, 918.7731934, -1494.8820801, 1490.8427734

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9448562
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9452548
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -703.1428223, 873.5878296, -734.7279663, 915.0056152, -1618.1484375, 1608.3157959
1: -511.0803223, 684.1479492, -536.9144287, 717.8038330, -1228.8840332, 1221.0623779
2: -436.8538208, 676.7639160, -458.7274780, 710.3518677, -1147.2056885, 1135.4914551
3: -611.4164429, 819.8616333, -642.1484375, 860.8912964, -1472.3076172, 1462.0100098
4: -578.5692749, 909.9353638, -607.7298584, 954.8682251, -1533.4373779, 1517.6649170

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9391766, upper bound: 1541.9432953
time: 0.75 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9391766, upper bound: 1541.9449630
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -877.4686279, 1099.2493896, -750.3693848, 935.8841553, -1813.3526611, 1849.6186523
1: -639.0079956, 860.6503906, -549.0753784, 734.3292847, -1373.3372803, 1409.7257080
2: -546.1976318, 852.3526001, -469.0012817, 726.9308472, -1273.1284180, 1321.3535156
3: -767.3111572, 1031.6871338, -656.8927612, 880.9337769, -1648.2448730, 1688.5797119
4: -724.3611450, 1146.1955566, -621.5335083, 977.2179565, -1701.5791016, 1767.7290039

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9290228
time: 0.89 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9298419
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -878.4260254, 1100.3701172, -750.3693848, 935.8841553, -1814.3101807, 1850.7395020
1: -639.6423950, 861.5164795, -549.0753784, 734.3292847, -1373.9716797, 1410.5917969
2: -546.7550659, 853.1940308, -469.0012817, 726.9308472, -1273.6857910, 1322.1953125
3: -768.1010742, 1032.6743164, -656.8927612, 880.9337769, -1649.0349121, 1689.5670166
4: -725.0787354, 1147.3596191, -621.5335083, 977.2179565, -1702.2966309, 1768.8930664

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9295863
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9324459
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -699.8526611, 870.4273682, -868.1961060, 1094.1484375, -1794.0008545, 1738.6235352
1: -508.8080750, 681.4273071, -639.7258301, 860.0143433, -1368.8222656, 1321.1527100
2: -434.8721313, 674.2592163, -546.0719604, 852.9528198, -1287.8249512, 1220.3310547
3: -608.8891602, 816.8630981, -766.8872070, 1033.7282715, -1642.6174316, 1583.7502441
4: -576.1090088, 906.6450806, -725.0382690, 1147.4240723, -1723.5329590, 1631.6833496

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9325510, upper bound: 1541.9375485
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9325511, upper bound: 1541.9404909
time: 0.83 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -877.4686279, 1099.2493896, -895.4052734, 1122.9678955, -2000.4365234, 1994.6546631
1: -639.0079956, 860.6503906, -653.3246460, 879.6784058, -1518.6864014, 1513.9748535
2: -546.1976318, 852.3526001, -558.1797485, 871.6618042, -1417.8592529, 1410.5319824
3: -767.3111572, 1031.6871338, -784.6664429, 1054.9698486, -1822.2810059, 1816.3535156
4: -724.3611450, 1146.1955566, -740.4485474, 1172.3004150, -1896.6616211, 1886.6440430

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9241006, upper bound: 1541.9264798
time: 1.02 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9241078, upper bound: 1541.9234154
time: 0.79 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -878.4260254, 1100.3701172, -895.4052734, 1122.9678955, -2001.3939209, 1995.7753906
1: -639.6423950, 861.5164795, -653.3246460, 879.6784058, -1519.3208008, 1514.8410645
2: -546.7550659, 853.1940308, -558.1797485, 871.6618042, -1418.4166260, 1411.3737793
3: -768.1010742, 1032.6743164, -784.6664429, 1054.9698486, -1823.0709229, 1817.3405762
4: -725.0787354, 1147.3596191, -740.4485474, 1172.3004150, -1897.3791504, 1887.8081055

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9241006, upper bound: 1541.9276971
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9241078, upper bound: 1541.9246969
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -697.6857910, 873.7763672, -699.8526611, 870.4273682, -1568.1130371, 1573.6290283
1: -515.0140991, 688.3702393, -508.8080750, 681.4273071, -1196.4411621, 1197.1781006
2: -439.6369324, 681.8969727, -434.8721313, 674.2592163, -1113.8961182, 1116.7690430
3: -615.4884033, 827.5158691, -608.8891602, 816.8630981, -1432.3510742, 1436.4050293
4: -583.1974487, 917.0775757, -576.1090088, 906.6450806, -1489.8425293, 1493.1865234

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9446024, upper bound: 1541.9421731
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9446024, upper bound: 1541.9421749
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -734.2792969, 914.4768677, -703.1428223, 873.5878296, -1607.8671875, 1617.6196289
1: -536.7048950, 717.4542847, -511.0803223, 684.1479492, -1220.8527832, 1228.5345459
2: -458.5287476, 710.0073242, -436.8538208, 676.7639160, -1135.2924805, 1146.8610840
3: -641.8637085, 860.5128174, -611.4164429, 819.8616333, -1461.7250977, 1471.9290771
4: -607.4680786, 954.3909912, -578.5692749, 909.9353638, -1517.4029541, 1532.9602051

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9409829, upper bound: 1541.9356442
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9409829, upper bound: 1541.9357183
time: 0.91 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -877.4686279, 1099.2493896, -1848.6148682, 1812.1402588
1: -548.4672241, 733.4370117, -639.0079956, 860.6503906, -1409.1175537, 1372.4450684
2: -468.4582825, 726.0541382, -546.1976318, 852.3526001, -1320.8104248, 1272.2515869
3: -656.1301880, 879.9159546, -767.3111572, 1031.6871338, -1687.8171387, 1647.2270508
4: -620.8171387, 976.0241699, -724.3611450, 1146.1955566, -1767.0126953, 1700.3852539

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9290016, upper bound: 1541.9334478
time: 0.80 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9298419, upper bound: 1541.9348460
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -749.3656616, 934.6717529, -878.4260254, 1100.3701172, -1849.7357178, 1813.0977783
1: -548.4672241, 733.4370117, -639.6423950, 861.5164795, -1409.9836426, 1373.0793457
2: -468.4582825, 726.0541382, -546.7550659, 853.1940308, -1321.6522217, 1272.8089600
3: -656.1301880, 879.9159546, -768.1010742, 1032.6743164, -1688.8044434, 1648.0170898
4: -620.8171387, 976.0241699, -725.0787354, 1147.3596191, -1768.1767578, 1701.1029053

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9290016, upper bound: 1541.9336531
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9298419, upper bound: 1541.9405715
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -877.4686279, 1099.2493896, -1993.5465088, 1999.0196533
1: -652.5545654, 878.5811157, -639.0079956, 860.6503906, -1513.2049561, 1517.5891113
2: -557.5065308, 870.5724487, -546.1976318, 852.3526001, -1409.8590088, 1416.7700195
3: -783.6868896, 1053.7010498, -767.3111572, 1031.6871338, -1815.3740234, 1821.0122070
4: -739.5656738, 1170.8101807, -724.3611450, 1146.1955566, -1885.7609863, 1895.1713867

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9300124, upper bound: 1541.9247423
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9234403, upper bound: 1541.9243774
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -894.2971802, 1121.5510254, -878.4260254, 1100.3701172, -1994.6672363, 1999.9770508
1: -652.5545654, 878.5811157, -639.6423950, 861.5164795, -1514.0710449, 1518.2235107
2: -557.5065308, 870.5724487, -546.7550659, 853.1940308, -1410.7005615, 1417.3275146
3: -783.6868896, 1053.7010498, -768.1010742, 1032.6743164, -1816.3612061, 1821.8021240
4: -739.5656738, 1170.8101807, -725.0787354, 1147.3596191, -1886.9250488, 1895.8889160

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9298927, upper bound: 1541.9247423
time: 0.93 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9310374, upper bound: 1541.9334100
time: 1.08 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9309251, upper bound: 1541.9315566
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -699.0020142, 875.4768677, -699.8526611, 870.4273682, -1569.4290771, 1575.3294678
1: -515.8696289, 689.6176147, -508.8080750, 681.4273071, -1197.2967529, 1198.4256592
2: -440.3870544, 683.1488037, -434.8721313, 674.2592163, -1114.6462402, 1118.0209961
3: -616.5712280, 828.9740601, -608.8891602, 816.8630981, -1433.4340820, 1437.8632812
4: -584.1976929, 918.7731934, -576.1090088, 906.6450806, -1490.8427734, 1494.8820801

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9448562, upper bound: 1541.9434031
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9448562, upper bound: 1541.9434036
time: 1.15 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -734.7279663, 915.0056152, -703.1428223, 873.5878296, -1608.3157959, 1618.1484375
1: -536.9144287, 717.8038330, -511.0803223, 684.1479492, -1221.0623779, 1228.8840332
2: -458.7274780, 710.3518677, -436.8538208, 676.7639160, -1135.4914551, 1147.2056885
3: -642.1484375, 860.8912964, -611.4164429, 819.8616333, -1462.0100098, 1472.3076172
4: -607.7298584, 954.8682251, -578.5692749, 909.9353638, -1517.6649170, 1533.4373779

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9432953, upper bound: 1541.9391766
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9432953, upper bound: 1541.9392271
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -877.4686279, 1099.2493896, -1849.6186523, 1813.3526611
1: -549.0753784, 734.3292847, -639.0079956, 860.6503906, -1409.7255859, 1373.3372803
2: -469.0012817, 726.9308472, -546.1976318, 852.3526001, -1321.3535156, 1273.1284180
3: -656.8927612, 880.9337769, -767.3111572, 1031.6871338, -1688.5797119, 1648.2448730
4: -621.5335083, 977.2179565, -724.3611450, 1146.1955566, -1767.7290039, 1701.5791016

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9290228, upper bound: 1541.9346210
time: 0.90 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9298661, upper bound: 1541.9360191
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -750.3693848, 935.8841553, -878.4260254, 1100.3701172, -1850.7395020, 1814.3101807
1: -549.0753784, 734.3292847, -639.6423950, 861.5164795, -1410.5917969, 1373.9716797
2: -469.0012817, 726.9308472, -546.7550659, 853.1940308, -1322.1953125, 1273.6857910
3: -656.8927612, 880.9337769, -768.1010742, 1032.6743164, -1689.5670166, 1649.0349121
4: -621.5335083, 977.2179565, -725.0787354, 1147.3596191, -1768.8930664, 1702.2966309

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9290228, upper bound: 1541.9347998
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9298661, upper bound: 1541.9416033
time: 0.91 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -868.1961060, 1094.1484375, -699.8526611, 870.4273682, -1738.6235352, 1794.0008545
1: -639.7258301, 860.0143433, -508.8080750, 681.4273071, -1321.1527100, 1368.8222656
2: -546.0719604, 852.9528198, -434.8721313, 674.2592163, -1220.3309326, 1287.8249512
3: -766.8872070, 1033.7282715, -608.8891602, 816.8630981, -1583.7502441, 1642.6174316
4: -725.0382690, 1147.4240723, -576.1090088, 906.6450806, -1631.6833496, 1723.5329590

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375485, upper bound: 1541.9325511
time: 0.79 seconds

## Relational analysis of NS_B2_A1_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9375485, upper bound: 1541.9326298
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -877.4686279, 1099.2493896, -1994.6546631, 2000.4365234
1: -653.3246460, 879.6784058, -639.0079956, 860.6503906, -1513.9748535, 1518.6864014
2: -558.1797485, 871.6618042, -546.1976318, 852.3526001, -1410.5319824, 1417.8592529
3: -784.6664429, 1054.9698486, -767.3111572, 1031.6871338, -1816.3533936, 1822.2810059
4: -740.4485474, 1172.3004150, -724.3611450, 1146.1955566, -1886.6440430, 1896.6616211

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_B2_B1_B1

### Relational analysis result of NS_B2_A1_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9303936, upper bound: 1541.9262978
time: 1.04 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9310889, upper bound: 1541.9352469
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9309301, upper bound: 1541.9323277
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -895.4052734, 1122.9678955, -878.4260254, 1100.3701172, -1995.7753906, 2001.3939209
1: -653.3246460, 879.6784058, -639.6423950, 861.5164795, -1514.8410645, 1519.3208008
2: -558.1797485, 871.6618042, -546.7550659, 853.1940308, -1411.3737793, 1418.4166260
3: -784.6664429, 1054.9698486, -768.1010742, 1032.6743164, -1817.3405762, 1823.0709229
4: -740.4485474, 1172.3004150, -725.0787354, 1147.3596191, -1887.8081055, 1897.3791504

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9303936, upper bound: 1541.9262988
time: 0.82 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9310889, upper bound: 1541.9352619
time: 1.25 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9309301, upper bound: 1541.9323279
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -718.3277588, 894.0231934, -719.8602295, 895.9888916, -1614.3166504, 1613.8831787
1: -522.8232422, 700.1860352, -523.8558350, 701.6317139, -1224.4545898, 1224.0418701
2: -446.7558289, 692.8474731, -447.6536560, 694.2944946, -1141.0502930, 1140.5006104
3: -625.7783813, 839.2131348, -627.0578003, 840.9013062, -1466.6796875, 1466.2709961
4: -591.8724976, 931.6036377, -593.0591431, 933.5846558, -1525.4571533, 1524.6625977

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9458219, upper bound: 1541.9456847
time: 1.12 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9458907, upper bound: 1541.9458914
time: 1.03 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -719.8602295, 895.9888916, -719.8602295, 895.9888916, -1615.8489990, 1615.8491211
1: -523.8558350, 701.6317139, -523.8558350, 701.6317139, -1225.4871826, 1225.4873047
2: -447.6536560, 694.2944946, -447.6536560, 694.2944946, -1141.9479980, 1141.9479980
3: -627.0578003, 840.9013062, -627.0578003, 840.9013062, -1467.9591064, 1467.9591064
4: -593.0591431, 933.5846558, -593.0591431, 933.5846558, -1526.6437988, 1526.6437988

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9458219, upper bound: 1541.9459801
time: 1.02 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9458907, upper bound: 1541.9461423
time: 1.15 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -699.8526611, 870.4273682, -856.8720093, 1078.3725586, -1778.2252197, 1727.2993164
1: -508.8080750, 681.4273071, -629.1713867, 846.1588135, -1354.9667969, 1310.5983887
2: -434.8721313, 674.2592163, -537.2572021, 839.0129395, -1273.8850098, 1211.5163574
3: -608.8891602, 816.8630981, -754.5156860, 1016.6870117, -1625.5761719, 1571.3784180
4: -576.1090088, 906.6450806, -713.0762939, 1128.6176758, -1704.7265625, 1619.7213135

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9375422
time: 0.82 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9404766
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -703.1428223, 873.5878296, -857.2614746, 1072.6735840, -1775.8164062, 1730.8493652
1: -511.0803223, 684.1479492, -623.4992676, 839.5472412, -1350.6275635, 1307.6470947
2: -436.8538208, 676.7639160, -533.0230103, 831.3569336, -1268.2106934, 1209.7867432
3: -611.4164429, 819.8616333, -748.4035034, 1006.2898560, -1617.7061768, 1568.2651367
4: -578.5692749, 909.9353638, -706.7315063, 1117.8292236, -1696.3981934, 1616.6665039

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9375422
time: 0.77 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9404766
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -878.4260254, 1100.3701172, -719.8602295, 895.9888916, -1774.4149170, 1820.2302246
1: -639.6423950, 861.5164795, -523.8558350, 701.6317139, -1341.2739258, 1385.3723145
2: -546.7550659, 853.1940308, -447.6536560, 694.2944946, -1241.0495605, 1300.8475342
3: -768.1010742, 1032.6743164, -627.0578003, 840.9013062, -1609.0024414, 1659.7321777
4: -725.0787354, 1147.3596191, -593.0591431, 933.5846558, -1658.6633301, 1740.4187012

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9341537, upper bound: 1541.9295863
time: 0.80 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9324436
time: 1.14 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -878.4260254, 1100.3701172, -878.4260254, 1100.3701172, -1978.7961426, 1978.7961426
1: -639.6423950, 861.5164795, -639.6423950, 861.5164795, -1501.1589355, 1501.1589355
2: -546.7550659, 853.1940308, -546.7550659, 853.1940308, -1399.9489746, 1399.9489746
3: -768.1010742, 1032.6743164, -768.1010742, 1032.6743164, -1800.7753906, 1800.7753906
4: -725.0787354, 1147.3596191, -725.0787354, 1147.3596191, -1872.4383545, 1872.4383545

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9272873, upper bound: 1541.9227746
time: 0.77 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1541.9295367, upper bound: 1541.9315908
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 7.59 seconds
NS_B1_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355307
NS_B1_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355307
NS_B1_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355309
NS_B1_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9388360, upper bound: 1541.9355307
NS_B1_A1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317885
NS_B1_A1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317892
NS_B1_A1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317885
NS_B1_A1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9376476, upper bound: 1541.9317893
NS_B1_A1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9324742, upper bound: 1541.9352828
NS_B1_A1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9324742, upper bound: 1541.9352828
NS_B1_A1_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9301761, upper bound: 1541.9247283
NS_B1_A1_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9244989, upper bound: 1541.9244661
NS_B1_A1_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9301761, upper bound: 1541.9247413
NS_B1_A1_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9244989, upper bound: 1541.9245197
NS_B1_A1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9434022
NS_B1_A1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9434022
NS_B1_A1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9436910
NS_B1_A1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9434022, upper bound: 1541.9436910
NS_B1_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9416095, upper bound: 1541.9401624
NS_B1_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9416095, upper bound: 1541.9401624
NS_B1_A1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9343570, upper bound: 1541.9353113
NS_B1_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9392565, upper bound: 1541.9374877
NS_B1_A1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9401624, upper bound: 1541.9416095
NS_B1_A1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9401624, upper bound: 1541.9416095
NS_B1_A1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9353112, upper bound: 1541.9343570
NS_B1_A1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9374877, upper bound: 1541.9392565
NS_B1_A1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9333753, upper bound: 1541.9302701
NS_B1_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9341200, upper bound: 1541.9375403
NS_B1_A1_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9315223, upper bound: 1541.9263113
NS_B1_A1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9245434, upper bound: 1541.9260528
NS_B1_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9446024
NS_B1_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9450699
NS_B1_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9356442, upper bound: 1541.9409829
NS_B1_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9356442, upper bound: 1541.9434154
NS_B1_A2_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9290016
NS_B1_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9298419
NS_B1_A2_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9295638
NS_B1_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9324271
NS_B1_A2_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9241006, upper bound: 1541.9243067
NS_B1_A2_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9241078, upper bound: 1541.9233540
NS_B1_A2_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9334097, upper bound: 1541.9314342
NS_B1_A2_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9315564, upper bound: 1541.9313221
NS_B1_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9448562
NS_B1_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9421731, upper bound: 1541.9452548
NS_B1_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9391766, upper bound: 1541.9432953
NS_B1_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9391766, upper bound: 1541.9449630
NS_B1_A2_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9290228
NS_B1_A2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9298419
NS_B1_A2_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9334478, upper bound: 1541.9295863
NS_B1_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9324459
NS_B1_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9325510, upper bound: 1541.9375485
NS_B1_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9325511, upper bound: 1541.9404909
NS_B1_A2_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9241006, upper bound: 1541.9264798
NS_B1_A2_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9241078, upper bound: 1541.9234154
NS_B1_A2_B2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9241006, upper bound: 1541.9276971
NS_B1_A2_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9241078, upper bound: 1541.9246969
NS_B2_A1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9446024, upper bound: 1541.9421731
NS_B2_A1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9446024, upper bound: 1541.9421749
NS_B2_A1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9409829, upper bound: 1541.9356442
NS_B2_A1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9409829, upper bound: 1541.9357183
NS_B2_A1_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9290016, upper bound: 1541.9334478
NS_B2_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9298419, upper bound: 1541.9348460
NS_B2_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9290016, upper bound: 1541.9336531
NS_B2_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9298419, upper bound: 1541.9405715
NS_B2_A1_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9300124, upper bound: 1541.9247423
NS_B2_A1_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9234403, upper bound: 1541.9243774
NS_B2_A1_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9310374, upper bound: 1541.9334100
NS_B2_A1_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9309251, upper bound: 1541.9315566
NS_B2_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9448562, upper bound: 1541.9434031
NS_B2_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9448562, upper bound: 1541.9434036
NS_B2_A1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9432953, upper bound: 1541.9391766
NS_B2_A1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9432953, upper bound: 1541.9392271
NS_B2_A1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9290228, upper bound: 1541.9346210
NS_B2_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9298661, upper bound: 1541.9360191
NS_B2_A1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9290228, upper bound: 1541.9347998
NS_B2_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9298661, upper bound: 1541.9416033
NS_B2_A1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9375485, upper bound: 1541.9325511
NS_B2_A1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9375485, upper bound: 1541.9326298
NS_B2_A1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9310889, upper bound: 1541.9352469
NS_B2_A1_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9309301, upper bound: 1541.9323277
NS_B2_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9310889, upper bound: 1541.9352619
NS_B2_A1_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9309301, upper bound: 1541.9323279
NS_B2_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9458219, upper bound: 1541.9456847
NS_B2_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9458907, upper bound: 1541.9458914
NS_B2_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9458219, upper bound: 1541.9459801
NS_B2_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9458907, upper bound: 1541.9461423
NS_B2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9375422
NS_B2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9404766
NS_B2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9375422
NS_B2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9315290, upper bound: 1541.9404766
NS_B2_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9341537, upper bound: 1541.9295863
NS_B2_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9348460, upper bound: 1541.9324436
NS_B2_A2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9272873, upper bound: 1541.9227746
NS_B2_A2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 7.59
Output dim: 0, lower bound: -1541.9295367, upper bound: 1541.9315908

## BFS NS instance: NS_B1_A1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -697.6857910, 873.7763672, -701.2141113, 878.2443848, -1575.9300537, 1574.9904785
1: -515.0140991, 688.3702393, -517.4451904, 691.7836914, -1206.7977295, 1205.8154297
2: -439.6369324, 681.8969727, -441.7453613, 685.2877808, -1124.9245605, 1123.6422119
3: -615.4884033, 827.5158691, -618.5140991, 831.5227051, -1447.0108643, 1446.0296631
4: -583.1974487, 917.0775757, -585.9918213, 921.6644287, -1504.8618164, 1503.0693359

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9392128, upper bound: 1541.9356064
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9392128, upper bound: 1541.9356264
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -734.2792969, 914.4768677, -701.2141113, 878.2443848, -1612.5236816, 1615.6909180
1: -536.7048950, 717.4542847, -517.4451904, 691.7836914, -1228.4885254, 1234.8994141
2: -458.5287476, 710.0073242, -441.7453613, 685.2877808, -1143.8162842, 1151.7526855
3: -641.8637085, 860.5128174, -618.5140991, 831.5227051, -1473.3864746, 1479.0263672
4: -607.4680786, 954.3909912, -585.9918213, 921.6644287, -1529.1323242, 1540.3828125

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9392128, upper bound: 1541.9356064
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9392128, upper bound: 1541.9356264
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -697.6857910, 873.7763672, -736.9458008, 917.7936401, -1615.4793701, 1610.7221680
1: -515.0140991, 688.3702393, -538.5001831, 719.9794922, -1234.9936523, 1226.8702393
2: -439.6369324, 681.8969727, -460.0958557, 712.5017090, -1152.1384277, 1141.9927979
3: -615.4884033, 827.5158691, -644.1045532, 863.4508057, -1478.9389648, 1471.6203613
4: -583.1974487, 917.0775757, -609.5385742, 957.7708130, -1540.9682617, 1526.6162109

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9354110, upper bound: 1541.9354110
time: 1.04 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9354110, upper bound: 1541.9355307
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -734.2792969, 914.4768677, -736.9458008, 917.7936401, -1652.0729980, 1651.4226074
1: -536.7048950, 717.4542847, -538.5001831, 719.9794922, -1256.6843262, 1255.9544678
2: -458.5287476, 710.0073242, -460.0958557, 712.5017090, -1171.0300293, 1170.1031494
3: -641.8637085, 860.5128174, -644.1045532, 863.4508057, -1505.3143311, 1504.6173096
4: -607.4680786, 954.3909912, -609.5385742, 957.7708130, -1565.2388916, 1563.9295654

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9354110, upper bound: 1541.9354110
time: 0.83 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9354110, upper bound: 1541.9355307
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -865.7438354, 1091.2446289, -701.2141113, 878.2443848, -1743.9882812, 1792.4587402
1: -638.0795288, 857.7741699, -517.4451904, 691.7836914, -1329.8629150, 1375.2193604
2: -544.6342163, 850.7475586, -441.7453613, 685.2877808, -1229.9218750, 1292.4929199
3: -764.8753052, 1031.1295166, -618.5140991, 831.5227051, -1596.3979492, 1649.6431885
4: -723.1636353, 1144.4539795, -585.9918213, 921.6644287, -1644.8281250, 1730.4458008

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9353880, upper bound: 1541.9335526
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9353880, upper bound: 1541.9336682
time: 0.95 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -865.7438354, 1091.2446289, -736.9458008, 917.7936401, -1783.5374756, 1828.1904297
1: -638.0795288, 857.7741699, -538.5001831, 719.9794922, -1358.0589600, 1396.2744141
2: -544.6342163, 850.7475586, -460.0958557, 712.5017090, -1257.1358643, 1310.8433838
3: -764.8753052, 1031.1295166, -644.1045532, 863.4508057, -1628.3261719, 1675.2341309
4: -723.1636353, 1144.4539795, -609.5385742, 957.7708130, -1680.9344482, 1753.9925537

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9353880, upper bound: 1541.9335526
time: 0.86 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1541.9353880, upper bound: 1541.9336682
time: 0.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.40 + 417.23 = 420.63 seconds
