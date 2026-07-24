## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01169311


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112354, 0.0112354)
1: (0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804)
2: (0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151616, 0.0151616)
3: (-0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664)
4: (-0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922)
5: (-0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806)
6: (-0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822)
7: (-0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923)
8: (-0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544)
9: (0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 1.92 = 3.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0137566, upper bound: 0.0137566

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135920, upper bound: 0.0125540
time: 1.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135922, upper bound: 0.0135922
time: 1.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.42 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 9, lower bound: -0.0135920, upper bound: 0.0125540
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 9, lower bound: -0.0135922, upper bound: 0.0135922

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056460, -0.0056143, 0.0058294, -0.0112039, 0.0109194
1: 0.0020836, 0.0126449, 0.0016540, 0.0125256, -0.0104420, 0.0109908
2: 0.0140275, 0.0311463, 0.0139574, 0.0311834, -0.0149115, 0.0149480
3: -0.0098849, -0.0019333, -0.0100549, -0.0020420, -0.0078429, 0.0081217
4: -0.0023122, 0.0024672, -0.0022878, 0.0028303, -0.0051424, 0.0047550
5: -0.0036777, 0.0074717, -0.0041416, 0.0073270, -0.0110046, 0.0116133
6: -0.0067717, -0.0005886, -0.0070374, -0.0006176, -0.0061541, 0.0064488
7: -0.0114247, 0.0002549, -0.0114443, 0.0007338, -0.0121585, 0.0116993
8: -0.0110869, 0.0008870, -0.0113094, 0.0007732, -0.0118601, 0.0121964
9: 0.9805632, 0.9978729, 0.9805619, 0.9988197, -0.0182565, 0.0173110

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132899, upper bound: 0.0123738
time: 1.13 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134335, upper bound: 0.0123738
time: 1.43 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0056211, 0.0058012, -0.0056372, 0.0058689, -0.0112626, 0.0111609
1: 0.0016580, 0.0125321, 0.0015671, 0.0125474, -0.0108894, 0.0109651
2: 0.0139484, 0.0311798, 0.0139266, 0.0312011, -0.0153717, 0.0150816
3: -0.0100363, -0.0020366, -0.0100901, -0.0020237, -0.0080126, 0.0080535
4: -0.0022891, 0.0028598, -0.0022926, 0.0028996, -0.0051886, 0.0051525
5: -0.0041294, 0.0073339, -0.0042298, 0.0073507, -0.0114802, 0.0115638
6: -0.0070354, -0.0006146, -0.0070884, -0.0006062, -0.0064292, 0.0064738
7: -0.0114481, 0.0007463, -0.0114622, 0.0008301, -0.0122782, 0.0122085
8: -0.0112797, 0.0007815, -0.0113532, 0.0008013, -0.0120810, 0.0121347
9: 0.9805568, 0.9988883, 0.9805366, 0.9990062, -0.0184494, 0.0183517

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135920
time: 1.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135922
time: 1.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.08 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 9, lower bound: -0.0132899, upper bound: 0.0123738
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 9, lower bound: -0.0134335, upper bound: 0.0123738
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135920
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.08
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135922

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056215, -0.0056427, 0.0056391, -0.0109729, 0.0108596
1: 0.0021106, 0.0126441, 0.0018589, 0.0125105, -0.0103999, 0.0107852
2: 0.0140277, 0.0311200, 0.0139379, 0.0309795, -0.0145697, 0.0147439
3: -0.0098707, -0.0019334, -0.0099457, -0.0020298, -0.0078409, 0.0080123
4: -0.0023121, 0.0024479, -0.0022715, 0.0026787, -0.0049908, 0.0047193
5: -0.0036436, 0.0074710, -0.0038758, 0.0073281, -0.0109717, 0.0113468
6: -0.0067542, -0.0005905, -0.0069040, -0.0006449, -0.0061093, 0.0063136
7: -0.0114159, 0.0002310, -0.0113493, 0.0005520, -0.0119679, 0.0115802
8: -0.0110619, 0.0008865, -0.0111179, 0.0007996, -0.0118616, 0.0120044
9: 0.9805752, 0.9978237, 0.9806947, 0.9984456, -0.0178704, 0.0171291

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
time: 1.25 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
time: 1.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056364, -0.0056142, 0.0056828, -0.0110287, 0.0108928
1: 0.0020930, 0.0126446, 0.0017968, 0.0125213, -0.0104283, 0.0108478
2: 0.0140275, 0.0311371, 0.0139580, 0.0310439, -0.0146469, 0.0149113
3: -0.0098795, -0.0019333, -0.0099727, -0.0020428, -0.0078367, 0.0080394
4: -0.0023121, 0.0024610, -0.0022876, 0.0027382, -0.0050503, 0.0047486
5: -0.0036653, 0.0074715, -0.0039536, 0.0073234, -0.0109887, 0.0114251
6: -0.0067658, -0.0005892, -0.0069470, -0.0006279, -0.0061379, 0.0063578
7: -0.0114218, 0.0002469, -0.0113985, 0.0006133, -0.0120351, 0.0116454
8: -0.0110769, 0.0008868, -0.0111561, 0.0007708, -0.0118477, 0.0120429
9: 0.9805671, 0.9978570, 0.9806260, 0.9985830, -0.0180159, 0.0172310

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
time: 1.24 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
time: 1.21 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056211, 0.0058012, -0.0057254, 0.0056460, -0.0109317, 0.0111854
1: 0.0016580, 0.0125321, 0.0020836, 0.0126449, -0.0109868, 0.0104485
2: 0.0139484, 0.0311798, 0.0140275, 0.0311463, -0.0149441, 0.0148952
3: -0.0100363, -0.0020366, -0.0098849, -0.0019333, -0.0081031, 0.0078483
4: -0.0022891, 0.0028598, -0.0023122, 0.0024672, -0.0047563, 0.0051720
5: -0.0041294, 0.0073339, -0.0036777, 0.0074717, -0.0116011, 0.0110116
6: -0.0070354, -0.0006146, -0.0067717, -0.0005886, -0.0064468, 0.0061571
7: -0.0114481, 0.0007463, -0.0114247, 0.0002549, -0.0117030, 0.0121710
8: -0.0112797, 0.0007815, -0.0110869, 0.0008870, -0.0121667, 0.0118684
9: 0.9805568, 0.9988883, 0.9805632, 0.9978729, -0.0173161, 0.0183251

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132899
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134335
time: 1.30 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056211, 0.0058012, -0.0056211, 0.0058012, -0.0111887, 0.0111887
1: 0.0016580, 0.0125321, 0.0016580, 0.0125321, -0.0108741, 0.0108741
2: 0.0139484, 0.0311798, 0.0139484, 0.0311798, -0.0153016, 0.0153016
3: -0.0100363, -0.0020366, -0.0100363, -0.0020366, -0.0079998, 0.0079998
4: -0.0022891, 0.0028598, -0.0022891, 0.0028598, -0.0051489, 0.0051489
5: -0.0041294, 0.0073339, -0.0041294, 0.0073339, -0.0114634, 0.0114634
6: -0.0070354, -0.0006146, -0.0070354, -0.0006146, -0.0064208, 0.0064208
7: -0.0114481, 0.0007463, -0.0114481, 0.0007463, -0.0121944, 0.0121944
8: -0.0112797, 0.0007815, -0.0112797, 0.0007815, -0.0120612, 0.0120612
9: 0.9805568, 0.9988883, 0.9805568, 0.9988883, -0.0183315, 0.0183315

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132910
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134342
time: 4.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.94 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132899
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134335
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132910
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.94
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134342

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056215, -0.0057575, 0.0054564, -0.0107583, 0.0109343
1: 0.0021106, 0.0126441, 0.0022875, 0.0126435, -0.0105329, 0.0103566
2: 0.0140277, 0.0311200, 0.0140086, 0.0309421, -0.0145179, 0.0146542
3: -0.0098707, -0.0019334, -0.0097759, -0.0019237, -0.0079470, 0.0078425
4: -0.0023121, 0.0024479, -0.0022995, 0.0023202, -0.0046323, 0.0047474
5: -0.0036436, 0.0074710, -0.0034181, 0.0074809, -0.0111246, 0.0108891
6: -0.0067542, -0.0005905, -0.0066403, -0.0006086, -0.0061456, 0.0060499
7: -0.0114159, 0.0002310, -0.0113376, 0.0000733, -0.0114892, 0.0115686
8: -0.0110619, 0.0008865, -0.0108967, 0.0009173, -0.0119792, 0.0117832
9: 0.9805752, 0.9978237, 0.9806842, 0.9975065, -0.0169313, 0.0171396

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
time: 1.11 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056215, -0.0056490, 0.0056098, -0.0109544, 0.0108579
1: 0.0021106, 0.0126441, 0.0018637, 0.0125166, -0.0104060, 0.0107803
2: 0.0140277, 0.0311200, 0.0139295, 0.0309755, -0.0145549, 0.0147160
3: -0.0098707, -0.0019334, -0.0099271, -0.0020247, -0.0078460, 0.0079936
4: -0.0023121, 0.0024479, -0.0022726, 0.0027076, -0.0050197, 0.0047205
5: -0.0036436, 0.0074710, -0.0038639, 0.0073346, -0.0109782, 0.0113349
6: -0.0067542, -0.0005905, -0.0069023, -0.0006421, -0.0061121, 0.0063119
7: -0.0114159, 0.0002310, -0.0113526, 0.0005621, -0.0119780, 0.0115836
8: -0.0110619, 0.0008865, -0.0110881, 0.0008073, -0.0118693, 0.0119746
9: 0.9805752, 0.9978237, 0.9806902, 0.9985070, -0.0179318, 0.0171335

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
time: 1.10 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056364, -0.0057253, 0.0054984, -0.0108114, 0.0109608
1: 0.0020930, 0.0126446, 0.0022284, 0.0126409, -0.0105479, 0.0104163
2: 0.0140275, 0.0311371, 0.0140281, 0.0310068, -0.0145948, 0.0148201
3: -0.0098795, -0.0019333, -0.0098028, -0.0019340, -0.0079456, 0.0078695
4: -0.0023121, 0.0024610, -0.0023119, 0.0023732, -0.0046854, 0.0047729
5: -0.0036653, 0.0074715, -0.0034860, 0.0074682, -0.0111335, 0.0109574
6: -0.0067658, -0.0005892, -0.0066810, -0.0005981, -0.0061677, 0.0060919
7: -0.0114218, 0.0002469, -0.0113801, 0.0001318, -0.0115537, 0.0116270
8: -0.0110769, 0.0008868, -0.0109331, 0.0008848, -0.0119617, 0.0118199
9: 0.9805671, 0.9978570, 0.9806254, 0.9976319, -0.0170648, 0.0172316

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
time: 1.14 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056364, -0.0056206, 0.0056552, -0.0110108, 0.0108908
1: 0.0020930, 0.0126446, 0.0017986, 0.0125274, -0.0104344, 0.0108460
2: 0.0140275, 0.0311371, 0.0139496, 0.0310403, -0.0146324, 0.0148815
3: -0.0098795, -0.0019333, -0.0099538, -0.0020377, -0.0078418, 0.0080205
4: -0.0023121, 0.0024610, -0.0022887, 0.0027693, -0.0050815, 0.0047497
5: -0.0036653, 0.0074715, -0.0039464, 0.0073299, -0.0109952, 0.0114179
6: -0.0067658, -0.0005892, -0.0069459, -0.0006251, -0.0061407, 0.0063567
7: -0.0114218, 0.0002469, -0.0114020, 0.0006266, -0.0120484, 0.0116489
8: -0.0110769, 0.0008868, -0.0111265, 0.0007786, -0.0118555, 0.0120133
9: 0.9805671, 0.9978570, 0.9806213, 0.9986537, -0.0180866, 0.0172357

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
time: 1.09 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0057254, 0.0056215, -0.0108791, 0.0109545
1: 0.0018636, 0.0125181, 0.0021106, 0.0126441, -0.0107805, 0.0104075
2: 0.0139272, 0.0309755, 0.0140277, 0.0311200, -0.0147753, 0.0145551
3: -0.0099271, -0.0020233, -0.0098707, -0.0019334, -0.0079937, 0.0078473
4: -0.0022731, 0.0027077, -0.0023121, 0.0024479, -0.0047210, 0.0050198
5: -0.0038639, 0.0073364, -0.0036436, 0.0074710, -0.0113349, 0.0109801
6: -0.0069024, -0.0006412, -0.0067542, -0.0005905, -0.0063119, 0.0061130
7: -0.0113537, 0.0005622, -0.0114159, 0.0002310, -0.0115847, 0.0119781
8: -0.0110881, 0.0008096, -0.0110619, 0.0008865, -0.0119746, 0.0118715
9: 0.9806883, 0.9985073, 0.9805752, 0.9978237, -0.0171354, 0.0179321

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0132899
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0132899
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0057254, 0.0056364, -0.0109069, 0.0110108
1: 0.0017986, 0.0125278, 0.0020930, 0.0126446, -0.0108461, 0.0104348
2: 0.0139490, 0.0310403, 0.0140275, 0.0311371, -0.0149015, 0.0146324
3: -0.0099538, -0.0020374, -0.0098795, -0.0019333, -0.0080205, 0.0078421
4: -0.0022889, 0.0027694, -0.0023121, 0.0024610, -0.0047498, 0.0050815
5: -0.0039464, 0.0073304, -0.0036653, 0.0074715, -0.0114179, 0.0109957
6: -0.0069459, -0.0006249, -0.0067658, -0.0005892, -0.0063567, 0.0061409
7: -0.0114023, 0.0006266, -0.0114218, 0.0002469, -0.0116492, 0.0120485
8: -0.0111265, 0.0007791, -0.0110769, 0.0008868, -0.0120133, 0.0118560
9: 0.9806208, 0.9986536, 0.9805671, 0.9978570, -0.0172362, 0.0180865

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0056211, 0.0057761, -0.0111346, 0.0109622
1: 0.0018636, 0.0125181, 0.0016855, 0.0125313, -0.0106677, 0.0108326
2: 0.0139272, 0.0309755, 0.0139486, 0.0311536, -0.0151231, 0.0149906
3: -0.0099271, -0.0020233, -0.0100221, -0.0020368, -0.0078903, 0.0079987
4: -0.0022731, 0.0027077, -0.0022890, 0.0028398, -0.0051129, 0.0049967
5: -0.0038639, 0.0073364, -0.0040944, 0.0073332, -0.0111971, 0.0114308
6: -0.0069024, -0.0006412, -0.0070177, -0.0006166, -0.0062858, 0.0063765
7: -0.0113537, 0.0005622, -0.0114391, 0.0007220, -0.0120757, 0.0120013
8: -0.0110881, 0.0008096, -0.0112546, 0.0007810, -0.0118692, 0.0120641
9: 0.9806883, 0.9985073, 0.9805694, 0.9988371, -0.0181488, 0.0179378

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0132910
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0132910
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0056211, 0.0057917, -0.0111654, 0.0110197
1: 0.0017986, 0.0125278, 0.0016672, 0.0125319, -0.0107333, 0.0108606
2: 0.0139490, 0.0310403, 0.0139484, 0.0311707, -0.0152721, 0.0150660
3: -0.0099538, -0.0020374, -0.0100309, -0.0020366, -0.0079172, 0.0079936
4: -0.0022889, 0.0027694, -0.0022891, 0.0028539, -0.0051428, 0.0050584
5: -0.0039464, 0.0073304, -0.0041174, 0.0073337, -0.0112801, 0.0114478
6: -0.0069459, -0.0006249, -0.0070295, -0.0006152, -0.0063306, 0.0064046
7: -0.0114023, 0.0006266, -0.0114451, 0.0007385, -0.0121408, 0.0120718
8: -0.0111265, 0.0007791, -0.0112697, 0.0007814, -0.0119079, 0.0120488
9: 0.9806208, 0.9986536, 0.9805609, 0.9988731, -0.0182523, 0.0180927

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342
time: 1.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.75 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123442
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0123738
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0132899
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0132899
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0132910
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0132910
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.75
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0057575, 0.0054564, -0.0107385, 0.0107385
1: 0.0022875, 0.0126435, 0.0022875, 0.0126435, -0.0103560, 0.0103560
2: 0.0140086, 0.0309421, 0.0140086, 0.0309421, -0.0143762, 0.0143762
3: -0.0097759, -0.0019237, -0.0097759, -0.0019237, -0.0078522, 0.0078522
4: -0.0022995, 0.0023202, -0.0022995, 0.0023202, -0.0046197, 0.0046197
5: -0.0034181, 0.0074809, -0.0034181, 0.0074809, -0.0108991, 0.0108991
6: -0.0066403, -0.0006086, -0.0066403, -0.0006086, -0.0060318, 0.0060318
7: -0.0113376, 0.0000733, -0.0113376, 0.0000733, -0.0114110, 0.0114110
8: -0.0108967, 0.0009173, -0.0108967, 0.0009173, -0.0118140, 0.0118140
9: 0.9806842, 0.9975065, 0.9806842, 0.9975065, -0.0168223, 0.0168223

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120407, upper bound: 0.0117398
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117294
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0057575, 0.0054564, -0.0107286, 0.0107892
1: 0.0022284, 0.0126409, 0.0022875, 0.0126435, -0.0104151, 0.0103534
2: 0.0140281, 0.0310068, 0.0140086, 0.0309421, -0.0143952, 0.0144459
3: -0.0098028, -0.0019340, -0.0097759, -0.0019237, -0.0078791, 0.0078419
4: -0.0023119, 0.0023732, -0.0022995, 0.0023202, -0.0046321, 0.0046727
5: -0.0034860, 0.0074682, -0.0034181, 0.0074809, -0.0109669, 0.0108864
6: -0.0066810, -0.0005981, -0.0066403, -0.0006086, -0.0060725, 0.0060422
7: -0.0113801, 0.0001318, -0.0113376, 0.0000733, -0.0114534, 0.0114695
8: -0.0109331, 0.0008848, -0.0108967, 0.0009173, -0.0118503, 0.0117815
9: 0.9806254, 0.9976319, 0.9806842, 0.9975065, -0.0168811, 0.0169477

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117386, upper bound: 0.0120866
time: 0.96 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0056490, 0.0056098, -0.0109346, 0.0106621
1: 0.0022875, 0.0126435, 0.0018637, 0.0125166, -0.0102291, 0.0107798
2: 0.0140086, 0.0309421, 0.0139295, 0.0309755, -0.0144131, 0.0144379
3: -0.0097759, -0.0019237, -0.0099271, -0.0020247, -0.0077512, 0.0080034
4: -0.0022995, 0.0023202, -0.0022726, 0.0027076, -0.0050071, 0.0045928
5: -0.0034181, 0.0074809, -0.0038639, 0.0073346, -0.0107527, 0.0113448
6: -0.0066403, -0.0006086, -0.0069023, -0.0006421, -0.0059983, 0.0062938
7: -0.0113376, 0.0000733, -0.0113526, 0.0005621, -0.0118997, 0.0114259
8: -0.0108967, 0.0009173, -0.0110881, 0.0008073, -0.0117041, 0.0120054
9: 0.9806842, 0.9975065, 0.9806902, 0.9985070, -0.0178229, 0.0168163

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117280
time: 1.26 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117208
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0056490, 0.0056098, -0.0109247, 0.0107128
1: 0.0022284, 0.0126409, 0.0018637, 0.0125166, -0.0102882, 0.0107771
2: 0.0140281, 0.0310068, 0.0139295, 0.0309755, -0.0144321, 0.0145077
3: -0.0098028, -0.0019340, -0.0099271, -0.0020247, -0.0077781, 0.0079931
4: -0.0023119, 0.0023732, -0.0022726, 0.0027076, -0.0050196, 0.0046459
5: -0.0034860, 0.0074682, -0.0038639, 0.0073346, -0.0108205, 0.0113321
6: -0.0066810, -0.0005981, -0.0069023, -0.0006421, -0.0060390, 0.0063042
7: -0.0113801, 0.0001318, -0.0113526, 0.0005621, -0.0119422, 0.0114844
8: -0.0109331, 0.0008848, -0.0110881, 0.0008073, -0.0117404, 0.0119729
9: 0.9806254, 0.9976319, 0.9806902, 0.9985070, -0.0178816, 0.0169417

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117568
time: 1.22 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0057253, 0.0054984, -0.0107892, 0.0107286
1: 0.0022875, 0.0126435, 0.0022284, 0.0126409, -0.0103534, 0.0104151
2: 0.0140086, 0.0309421, 0.0140281, 0.0310068, -0.0144459, 0.0143952
3: -0.0097759, -0.0019237, -0.0098028, -0.0019340, -0.0078419, 0.0078791
4: -0.0022995, 0.0023202, -0.0023119, 0.0023732, -0.0046727, 0.0046321
5: -0.0034181, 0.0074809, -0.0034860, 0.0074682, -0.0108864, 0.0109669
6: -0.0066403, -0.0006086, -0.0066810, -0.0005981, -0.0060422, 0.0060725
7: -0.0113376, 0.0000733, -0.0113801, 0.0001318, -0.0114695, 0.0114534
8: -0.0108967, 0.0009173, -0.0109331, 0.0008848, -0.0117815, 0.0118503
9: 0.9806842, 0.9975065, 0.9806254, 0.9976319, -0.0169477, 0.0168811

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120407, upper bound: 0.0117386
time: 1.04 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117272
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0057253, 0.0054984, -0.0108014, 0.0108014
1: 0.0022284, 0.0126409, 0.0022284, 0.0126409, -0.0104125, 0.0104125
2: 0.0140281, 0.0310068, 0.0140281, 0.0310068, -0.0146012, 0.0146012
3: -0.0098028, -0.0019340, -0.0098028, -0.0019340, -0.0078688, 0.0078688
4: -0.0023119, 0.0023732, -0.0023119, 0.0023732, -0.0046852, 0.0046852
5: -0.0034860, 0.0074682, -0.0034860, 0.0074682, -0.0109542, 0.0109542
6: -0.0066810, -0.0005981, -0.0066810, -0.0005981, -0.0060829, 0.0060829
7: -0.0113801, 0.0001318, -0.0113801, 0.0001318, -0.0115119, 0.0115119
8: -0.0109331, 0.0008848, -0.0109331, 0.0008848, -0.0118178, 0.0118178
9: 0.9806254, 0.9976319, 0.9806254, 0.9976319, -0.0170065, 0.0170065

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120407, upper bound: 0.0117667
time: 1.03 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0056206, 0.0056552, -0.0109886, 0.0106575
1: 0.0022875, 0.0126435, 0.0017986, 0.0125274, -0.0102399, 0.0108449
2: 0.0140086, 0.0309421, 0.0139496, 0.0310403, -0.0144835, 0.0144571
3: -0.0097759, -0.0019237, -0.0099538, -0.0020377, -0.0077382, 0.0080302
4: -0.0022995, 0.0023202, -0.0022887, 0.0027693, -0.0050689, 0.0046089
5: -0.0034181, 0.0074809, -0.0039464, 0.0073299, -0.0107480, 0.0114274
6: -0.0066403, -0.0006086, -0.0069459, -0.0006251, -0.0060152, 0.0063373
7: -0.0113376, 0.0000733, -0.0114020, 0.0006266, -0.0119642, 0.0114753
8: -0.0108967, 0.0009173, -0.0111265, 0.0007786, -0.0116753, 0.0120438
9: 0.9806842, 0.9975065, 0.9806213, 0.9986537, -0.0179695, 0.0168852

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117276
time: 1.20 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117195
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0056206, 0.0056552, -0.0110013, 0.0107314
1: 0.0022284, 0.0126409, 0.0017986, 0.0125274, -0.0102990, 0.0108423
2: 0.0140281, 0.0310068, 0.0139496, 0.0310403, -0.0146369, 0.0146626
3: -0.0098028, -0.0019340, -0.0099538, -0.0020377, -0.0077651, 0.0080199
4: -0.0023119, 0.0023732, -0.0022887, 0.0027693, -0.0050813, 0.0046619
5: -0.0034860, 0.0074682, -0.0039464, 0.0073299, -0.0108159, 0.0114146
6: -0.0066810, -0.0005981, -0.0069459, -0.0006251, -0.0060559, 0.0063477
7: -0.0113801, 0.0001318, -0.0114020, 0.0006266, -0.0120067, 0.0115338
8: -0.0109331, 0.0008848, -0.0111265, 0.0007786, -0.0117116, 0.0120113
9: 0.9806254, 0.9976319, 0.9806213, 0.9986537, -0.0180283, 0.0170106

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117568
time: 1.35 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0057575, 0.0054564, -0.0106869, 0.0109347
1: 0.0018636, 0.0125181, 0.0022875, 0.0126435, -0.0107799, 0.0102306
2: 0.0139272, 0.0309755, 0.0140086, 0.0309421, -0.0145113, 0.0144133
3: -0.0099271, -0.0020233, -0.0097759, -0.0019237, -0.0080034, 0.0077525
4: -0.0022731, 0.0027077, -0.0022995, 0.0023202, -0.0045932, 0.0050072
5: -0.0038639, 0.0073364, -0.0034181, 0.0074809, -0.0113449, 0.0107546
6: -0.0069024, -0.0006412, -0.0066403, -0.0006086, -0.0062938, 0.0059991
7: -0.0113537, 0.0005622, -0.0113376, 0.0000733, -0.0114270, 0.0118998
8: -0.0110881, 0.0008096, -0.0108967, 0.0009173, -0.0120054, 0.0117063
9: 0.9806883, 0.9985073, 0.9806842, 0.9975065, -0.0168182, 0.0178231

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117280, upper bound: 0.0130470
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117208, upper bound: 0.0128989
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0057253, 0.0054984, -0.0107385, 0.0109248
1: 0.0018636, 0.0125181, 0.0022284, 0.0126409, -0.0107773, 0.0102898
2: 0.0139272, 0.0309755, 0.0140281, 0.0310068, -0.0145801, 0.0144323
3: -0.0099271, -0.0020233, -0.0098028, -0.0019340, -0.0079931, 0.0077795
4: -0.0022731, 0.0027077, -0.0023119, 0.0023732, -0.0046463, 0.0050196
5: -0.0038639, 0.0073364, -0.0034860, 0.0074682, -0.0113322, 0.0108224
6: -0.0069024, -0.0006412, -0.0066810, -0.0005981, -0.0063043, 0.0060398
7: -0.0113537, 0.0005622, -0.0113801, 0.0001318, -0.0114855, 0.0119423
8: -0.0110881, 0.0008096, -0.0109331, 0.0008848, -0.0119729, 0.0117426
9: 0.9806883, 0.9985073, 0.9806254, 0.9976319, -0.0169436, 0.0178819

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117280, upper bound: 0.0130470
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117208, upper bound: 0.0128989
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0057575, 0.0054564, -0.0106802, 0.0109886
1: 0.0017986, 0.0125278, 0.0022875, 0.0126435, -0.0108449, 0.0102403
2: 0.0139490, 0.0310403, 0.0140086, 0.0309421, -0.0145257, 0.0144835
3: -0.0099538, -0.0020374, -0.0097759, -0.0019237, -0.0080302, 0.0077385
4: -0.0022889, 0.0027694, -0.0022995, 0.0023202, -0.0046090, 0.0050689
5: -0.0039464, 0.0073304, -0.0034181, 0.0074809, -0.0114274, 0.0107485
6: -0.0069459, -0.0006249, -0.0066403, -0.0006086, -0.0063373, 0.0060154
7: -0.0114023, 0.0006266, -0.0113376, 0.0000733, -0.0114756, 0.0119643
8: -0.0111265, 0.0007791, -0.0108967, 0.0009173, -0.0120438, 0.0116758
9: 0.9806208, 0.9986536, 0.9806842, 0.9975065, -0.0168857, 0.0179694

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117276, upper bound: 0.0131797
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117195, upper bound: 0.0130346
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0057253, 0.0054984, -0.0107508, 0.0110013
1: 0.0017986, 0.0125278, 0.0022284, 0.0126409, -0.0108423, 0.0102994
2: 0.0139490, 0.0310403, 0.0140281, 0.0310068, -0.0146953, 0.0146370
3: -0.0099538, -0.0020374, -0.0098028, -0.0019340, -0.0080199, 0.0077654
4: -0.0022889, 0.0027694, -0.0023119, 0.0023732, -0.0046621, 0.0050813
5: -0.0039464, 0.0073304, -0.0034860, 0.0074682, -0.0114147, 0.0108164
6: -0.0069459, -0.0006249, -0.0066810, -0.0005981, -0.0063478, 0.0060561
7: -0.0114023, 0.0006266, -0.0113801, 0.0001318, -0.0115341, 0.0120067
8: -0.0111265, 0.0007791, -0.0109331, 0.0008848, -0.0120113, 0.0117122
9: 0.9806208, 0.9986536, 0.9806254, 0.9976319, -0.0170111, 0.0180282

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117276, upper bound: 0.0131797
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117195, upper bound: 0.0130346
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0056508, 0.0056098, -0.0109416, 0.0109416
1: 0.0018636, 0.0125181, 0.0018636, 0.0125181, -0.0106545, 0.0106545
2: 0.0139272, 0.0309755, 0.0139272, 0.0309755, -0.0148675, 0.0148675
3: -0.0099271, -0.0020233, -0.0099271, -0.0020233, -0.0079038, 0.0079038
4: -0.0022731, 0.0027077, -0.0022731, 0.0027077, -0.0049808, 0.0049808
5: -0.0038639, 0.0073364, -0.0038639, 0.0073364, -0.0112004, 0.0112004
6: -0.0069024, -0.0006412, -0.0069024, -0.0006412, -0.0062611, 0.0062611
7: -0.0113537, 0.0005622, -0.0113537, 0.0005622, -0.0119159, 0.0119159
8: -0.0110881, 0.0008096, -0.0110881, 0.0008096, -0.0118977, 0.0118977
9: 0.9806883, 0.9985073, 0.9806883, 0.9985073, -0.0178189, 0.0178189

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122388, upper bound: 0.0129305
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121058, upper bound: 0.0129305
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0056210, 0.0056552, -0.0109969, 0.0109361
1: 0.0018636, 0.0125181, 0.0017986, 0.0125278, -0.0106642, 0.0107196
2: 0.0139272, 0.0309755, 0.0139490, 0.0310403, -0.0149371, 0.0148869
3: -0.0099271, -0.0020233, -0.0099538, -0.0020374, -0.0078897, 0.0079305
4: -0.0022731, 0.0027077, -0.0022889, 0.0027694, -0.0050424, 0.0049965
5: -0.0038639, 0.0073364, -0.0039464, 0.0073304, -0.0111943, 0.0112829
6: -0.0069024, -0.0006412, -0.0069459, -0.0006249, -0.0062775, 0.0063046
7: -0.0113537, 0.0005622, -0.0114023, 0.0006266, -0.0119803, 0.0119645
8: -0.0110881, 0.0008096, -0.0111265, 0.0007791, -0.0118673, 0.0119361
9: 0.9806883, 0.9985073, 0.9806208, 0.9986536, -0.0179653, 0.0178865

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122388, upper bound: 0.0129305
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121058, upper bound: 0.0129305
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0056508, 0.0056098, -0.0109361, 0.0109969
1: 0.0017986, 0.0125278, 0.0018636, 0.0125181, -0.0107196, 0.0106642
2: 0.0139490, 0.0310403, 0.0139272, 0.0309755, -0.0148869, 0.0149371
3: -0.0099538, -0.0020374, -0.0099271, -0.0020233, -0.0079305, 0.0078897
4: -0.0022889, 0.0027694, -0.0022731, 0.0027077, -0.0049965, 0.0050424
5: -0.0039464, 0.0073304, -0.0038639, 0.0073364, -0.0112829, 0.0111943
6: -0.0069459, -0.0006249, -0.0069024, -0.0006412, -0.0063046, 0.0062775
7: -0.0114023, 0.0006266, -0.0113537, 0.0005622, -0.0119645, 0.0119803
8: -0.0111265, 0.0007791, -0.0110881, 0.0008096, -0.0119361, 0.0118673
9: 0.9806208, 0.9986536, 0.9806883, 0.9985073, -0.0178865, 0.0179653

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0131878
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0056210, 0.0056552, -0.0110116, 0.0110116
1: 0.0017986, 0.0125278, 0.0017986, 0.0125278, -0.0107292, 0.0107292
2: 0.0139490, 0.0310403, 0.0139490, 0.0310403, -0.0150688, 0.0150688
3: -0.0099538, -0.0020374, -0.0099538, -0.0020374, -0.0079165, 0.0079165
4: -0.0022889, 0.0027694, -0.0022889, 0.0027694, -0.0050582, 0.0050582
5: -0.0039464, 0.0073304, -0.0039464, 0.0073304, -0.0112768, 0.0112768
6: -0.0069459, -0.0006249, -0.0069459, -0.0006249, -0.0063210, 0.0063210
7: -0.0114023, 0.0006266, -0.0114023, 0.0006266, -0.0120289, 0.0120289
8: -0.0111265, 0.0007791, -0.0111265, 0.0007791, -0.0119056, 0.0119056
9: 0.9806208, 0.9986536, 0.9806208, 0.9986536, -0.0180328, 0.0180328

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122381, upper bound: 0.0130680
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
time: 1.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.82 seconds
NS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0120407, upper bound: 0.0117398
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117294
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117386, upper bound: 0.0120866
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
NS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117280
NS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117208
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117568
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0120407, upper bound: 0.0117386
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117272
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0120407, upper bound: 0.0117667
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117276
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117195
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0130470, upper bound: 0.0117568
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117280, upper bound: 0.0130470
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117208, upper bound: 0.0128989
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117280, upper bound: 0.0130470
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117208, upper bound: 0.0128989
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117276, upper bound: 0.0131797
NS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117195, upper bound: 0.0130346
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117276, upper bound: 0.0131797
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0117195, upper bound: 0.0130346
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0122388, upper bound: 0.0129305
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0121058, upper bound: 0.0129305
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0122388, upper bound: 0.0129305
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0121058, upper bound: 0.0129305
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0131878
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0122381, upper bound: 0.0130680
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.82
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680

## BFS NS instance: NS_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0057575, 0.0054564, -0.0105441, 0.0107162
1: 0.0023100, 0.0124183, 0.0022875, 0.0126435, -0.0103335, 0.0101308
2: 0.0142510, 0.0309143, 0.0140086, 0.0309421, -0.0141511, 0.0143269
3: -0.0097684, -0.0021151, -0.0097759, -0.0019237, -0.0078447, 0.0076608
4: -0.0022332, 0.0023064, -0.0022995, 0.0023202, -0.0045533, 0.0046059
5: -0.0034008, 0.0072732, -0.0034181, 0.0074809, -0.0108817, 0.0106914
6: -0.0066324, -0.0007213, -0.0066403, -0.0006086, -0.0060238, 0.0059191
7: -0.0111300, 0.0000567, -0.0113376, 0.0000733, -0.0112034, 0.0113943
8: -0.0108863, 0.0006757, -0.0108967, 0.0009173, -0.0118035, 0.0115725
9: 0.9809389, 0.9974611, 0.9806842, 0.9975065, -0.0165676, 0.0167769

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117294, upper bound: 0.0117294
time: 1.00 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117294, upper bound: 0.0117294
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0056442, 0.0054494, -0.0101633, 0.0111234
1: 0.0019422, 0.0118258, 0.0023000, 0.0124844, -0.0105422, 0.0095257
2: 0.0146924, 0.0316475, 0.0141327, 0.0309296, -0.0139304, 0.0149092
3: -0.0101021, -0.0024892, -0.0097721, -0.0020284, -0.0080737, 0.0072829
4: -0.0020490, 0.0024703, -0.0022531, 0.0023132, -0.0043623, 0.0047234
5: -0.0037235, 0.0067224, -0.0034084, 0.0073350, -0.0110584, 0.0101309
6: -0.0068225, -0.0009472, -0.0066358, -0.0006839, -0.0061386, 0.0056887
7: -0.0110654, 0.0004304, -0.0112245, 0.0000640, -0.0111294, 0.0116549
8: -0.0112907, 0.0001849, -0.0108914, 0.0007834, -0.0120741, 0.0110763
9: 0.9809808, 0.9978092, 0.9808201, 0.9974815, -0.0165007, 0.0169892

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112211, upper bound: 0.0111377
time: 0.98 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117230, upper bound: 0.0117230
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0055529, 0.0054417, -0.0107064, 0.0105948
1: 0.0022284, 0.0126409, 0.0023100, 0.0124183, -0.0101899, 0.0103308
2: 0.0140281, 0.0310068, 0.0142510, 0.0309143, -0.0143459, 0.0142209
3: -0.0098028, -0.0019340, -0.0097684, -0.0021151, -0.0076877, 0.0078345
4: -0.0023119, 0.0023732, -0.0022332, 0.0023064, -0.0046184, 0.0046064
5: -0.0034860, 0.0074682, -0.0034008, 0.0072732, -0.0107592, 0.0108690
6: -0.0066810, -0.0005981, -0.0066324, -0.0007213, -0.0059598, 0.0060343
7: -0.0113801, 0.0001318, -0.0111300, 0.0000567, -0.0114368, 0.0112619
8: -0.0109331, 0.0008848, -0.0108863, 0.0006757, -0.0116088, 0.0117710
9: 0.9806254, 0.9976319, 0.9809389, 0.9974611, -0.0168357, 0.0166930

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056117, 0.0054914, -0.0051373, 0.0059815, -0.0111142, 0.0102140
1: 0.0022405, 0.0124835, 0.0019422, 0.0118258, -0.0095852, 0.0105413
2: 0.0141508, 0.0309941, 0.0146924, 0.0316475, -0.0149287, 0.0140004
3: -0.0097989, -0.0020399, -0.0101021, -0.0024892, -0.0073097, 0.0080622
4: -0.0022650, 0.0023663, -0.0020490, 0.0024703, -0.0047353, 0.0044154
5: -0.0034764, 0.0073230, -0.0037235, 0.0067224, -0.0101988, 0.0110464
6: -0.0066766, -0.0006739, -0.0068225, -0.0009472, -0.0057294, 0.0061486
7: -0.0112690, 0.0001226, -0.0110654, 0.0004304, -0.0116994, 0.0111880
8: -0.0109277, 0.0007509, -0.0112907, 0.0001849, -0.0111125, 0.0120416
9: 0.9807613, 0.9976075, 0.9809808, 0.9978092, -0.0170479, 0.0166268

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0112609
time: 0.95 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117210, upper bound: 0.0117562
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0056490, 0.0056098, -0.0107402, 0.0106398
1: 0.0023100, 0.0124183, 0.0018637, 0.0125166, -0.0102066, 0.0105545
2: 0.0142510, 0.0309143, 0.0139295, 0.0309755, -0.0141881, 0.0143887
3: -0.0097684, -0.0021151, -0.0099271, -0.0020247, -0.0077437, 0.0078120
4: -0.0022332, 0.0023064, -0.0022726, 0.0027076, -0.0049408, 0.0045791
5: -0.0034008, 0.0072732, -0.0038639, 0.0073346, -0.0107354, 0.0111371
6: -0.0066324, -0.0007213, -0.0069023, -0.0006421, -0.0059903, 0.0061811
7: -0.0111300, 0.0000567, -0.0113526, 0.0005621, -0.0116921, 0.0114093
8: -0.0108863, 0.0006757, -0.0110881, 0.0008073, -0.0116936, 0.0117639
9: 0.9809389, 0.9974611, 0.9806902, 0.9985070, -0.0175681, 0.0167709

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117208
time: 1.26 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117208
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0055368, 0.0056024, -0.0103595, 0.0110446
1: 0.0019422, 0.0118258, 0.0018770, 0.0123605, -0.0104183, 0.0099487
2: 0.0146924, 0.0316475, 0.0140542, 0.0309614, -0.0139676, 0.0149705
3: -0.0101021, -0.0024892, -0.0099233, -0.0021301, -0.0079720, 0.0074341
4: -0.0020490, 0.0024703, -0.0022269, 0.0027004, -0.0047495, 0.0046972
5: -0.0037235, 0.0067224, -0.0038543, 0.0071903, -0.0109138, 0.0105768
6: -0.0068225, -0.0009472, -0.0068976, -0.0007170, -0.0061055, 0.0059504
7: -0.0110654, 0.0004304, -0.0112404, 0.0005519, -0.0116173, 0.0116709
8: -0.0112907, 0.0001849, -0.0110828, 0.0006721, -0.0119628, 0.0112677
9: 0.9809808, 0.9978092, 0.9808258, 0.9984816, -0.0175009, 0.0169834

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0111377
time: 1.12 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117141
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0056490, 0.0056098, -0.0107284, 0.0106902
1: 0.0022505, 0.0124151, 0.0018637, 0.0125166, -0.0102661, 0.0105514
2: 0.0142708, 0.0309785, 0.0139295, 0.0309755, -0.0142066, 0.0144586
3: -0.0097951, -0.0021241, -0.0099271, -0.0020247, -0.0077704, 0.0078030
4: -0.0022445, 0.0023594, -0.0022726, 0.0027076, -0.0049522, 0.0046320
5: -0.0034685, 0.0072581, -0.0038639, 0.0073346, -0.0108031, 0.0111220
6: -0.0066730, -0.0007098, -0.0069023, -0.0006421, -0.0060309, 0.0061925
7: -0.0111719, 0.0001152, -0.0113526, 0.0005621, -0.0117340, 0.0114678
8: -0.0109222, 0.0006468, -0.0110881, 0.0008073, -0.0117296, 0.0117350
9: 0.9808817, 0.9975876, 0.9806902, 0.9985070, -0.0176253, 0.0168974

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
time: 1.85 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0055368, 0.0056024, -0.0103514, 0.0111003
1: 0.0018756, 0.0118278, 0.0018770, 0.0123605, -0.0104849, 0.0099507
2: 0.0147073, 0.0317128, 0.0140542, 0.0309614, -0.0139866, 0.0150425
3: -0.0101313, -0.0025013, -0.0099233, -0.0021301, -0.0080012, 0.0074219
4: -0.0020544, 0.0025248, -0.0022269, 0.0027004, -0.0047548, 0.0047517
5: -0.0038068, 0.0067076, -0.0038543, 0.0071903, -0.0109971, 0.0105619
6: -0.0068669, -0.0009356, -0.0068976, -0.0007170, -0.0061499, 0.0059620
7: -0.0111134, 0.0004922, -0.0112404, 0.0005519, -0.0116653, 0.0117327
8: -0.0113331, 0.0001610, -0.0110828, 0.0006721, -0.0120051, 0.0112438
9: 0.9809233, 0.9979489, 0.9808258, 0.9984816, -0.0175583, 0.0171230

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123607, upper bound: 0.0111694
time: 1.10 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0057253, 0.0054984, -0.0105948, 0.0107064
1: 0.0023100, 0.0124183, 0.0022284, 0.0126409, -0.0103308, 0.0101899
2: 0.0142510, 0.0309143, 0.0140281, 0.0310068, -0.0142209, 0.0143459
3: -0.0097684, -0.0021151, -0.0098028, -0.0019340, -0.0078345, 0.0076877
4: -0.0022332, 0.0023064, -0.0023119, 0.0023732, -0.0046064, 0.0046184
5: -0.0034008, 0.0072732, -0.0034860, 0.0074682, -0.0108690, 0.0107592
6: -0.0066324, -0.0007213, -0.0066810, -0.0005981, -0.0060343, 0.0059598
7: -0.0111300, 0.0000567, -0.0113801, 0.0001318, -0.0112619, 0.0114368
8: -0.0108863, 0.0006757, -0.0109331, 0.0008848, -0.0117710, 0.0116088
9: 0.9809389, 0.9974611, 0.9806254, 0.9976319, -0.0166930, 0.0168357

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117597, upper bound: 0.0117272
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117597, upper bound: 0.0117272
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0056117, 0.0054914, -0.0102140, 0.0111142
1: 0.0019422, 0.0118258, 0.0022405, 0.0124835, -0.0105413, 0.0095852
2: 0.0146924, 0.0316475, 0.0141508, 0.0309941, -0.0140004, 0.0149287
3: -0.0101021, -0.0024892, -0.0097989, -0.0020399, -0.0080622, 0.0073097
4: -0.0020490, 0.0024703, -0.0022650, 0.0023663, -0.0044154, 0.0047353
5: -0.0037235, 0.0067224, -0.0034764, 0.0073230, -0.0110464, 0.0101988
6: -0.0068225, -0.0009472, -0.0066766, -0.0006739, -0.0061486, 0.0057294
7: -0.0110654, 0.0004304, -0.0112690, 0.0001226, -0.0111880, 0.0116994
8: -0.0112907, 0.0001849, -0.0109277, 0.0007509, -0.0120416, 0.0111125
9: 0.9809808, 0.9978092, 0.9807613, 0.9976075, -0.0166268, 0.0170479

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112609, upper bound: 0.0111372
time: 0.99 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117562, upper bound: 0.0117210
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0057253, 0.0054984, -0.0106049, 0.0107787
1: 0.0022505, 0.0124151, 0.0022284, 0.0126409, -0.0103904, 0.0101868
2: 0.0142708, 0.0309785, 0.0140281, 0.0310068, -0.0143762, 0.0145493
3: -0.0097951, -0.0021241, -0.0098028, -0.0019340, -0.0078611, 0.0076787
4: -0.0022445, 0.0023594, -0.0023119, 0.0023732, -0.0046178, 0.0046714
5: -0.0034685, 0.0072581, -0.0034860, 0.0074682, -0.0109368, 0.0107441
6: -0.0066730, -0.0007098, -0.0066810, -0.0005981, -0.0060749, 0.0059713
7: -0.0111719, 0.0001152, -0.0113801, 0.0001318, -0.0113037, 0.0114952
8: -0.0109222, 0.0006468, -0.0109331, 0.0008848, -0.0118070, 0.0115799
9: 0.9808817, 0.9975876, 0.9806254, 0.9976319, -0.0167502, 0.0169622

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117278, upper bound: 0.0117597
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117278, upper bound: 0.0117597
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0056117, 0.0054914, -0.0102272, 0.0111938
1: 0.0018756, 0.0118278, 0.0022405, 0.0124835, -0.0106078, 0.0095873
2: 0.0147073, 0.0317128, 0.0141508, 0.0309941, -0.0141492, 0.0151433
3: -0.0101313, -0.0025013, -0.0097989, -0.0020399, -0.0080914, 0.0072976
4: -0.0020544, 0.0025248, -0.0022650, 0.0023663, -0.0044208, 0.0047898
5: -0.0038068, 0.0067076, -0.0034764, 0.0073230, -0.0111297, 0.0101840
6: -0.0068669, -0.0009356, -0.0066766, -0.0006739, -0.0061931, 0.0057410
7: -0.0111134, 0.0004922, -0.0112690, 0.0001226, -0.0112360, 0.0117612
8: -0.0113331, 0.0001610, -0.0109277, 0.0007509, -0.0120840, 0.0110886
9: 0.9809233, 0.9979489, 0.9807613, 0.9976075, -0.0166842, 0.0171876

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112204, upper bound: 0.0111694
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117215, upper bound: 0.0117562
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0056206, 0.0056552, -0.0107942, 0.0106352
1: 0.0023100, 0.0124183, 0.0017986, 0.0125274, -0.0102174, 0.0106197
2: 0.0142510, 0.0309143, 0.0139496, 0.0310403, -0.0142585, 0.0144079
3: -0.0097684, -0.0021151, -0.0099538, -0.0020377, -0.0077307, 0.0078388
4: -0.0022332, 0.0023064, -0.0022887, 0.0027693, -0.0050025, 0.0045951
5: -0.0034008, 0.0072732, -0.0039464, 0.0073299, -0.0107307, 0.0112197
6: -0.0066324, -0.0007213, -0.0069459, -0.0006251, -0.0060073, 0.0062246
7: -0.0111300, 0.0000567, -0.0114020, 0.0006266, -0.0117566, 0.0114587
8: -0.0108863, 0.0006757, -0.0111265, 0.0007786, -0.0116648, 0.0118023
9: 0.9809389, 0.9974611, 0.9806213, 0.9986537, -0.0177147, 0.0168398

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117195
time: 1.10 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117195
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0055071, 0.0056478, -0.0104133, 0.0110405
1: 0.0019422, 0.0118258, 0.0018118, 0.0123719, -0.0104298, 0.0100139
2: 0.0146924, 0.0316475, 0.0140726, 0.0310262, -0.0140382, 0.0149901
3: -0.0101021, -0.0024892, -0.0099500, -0.0021437, -0.0079585, 0.0074609
4: -0.0020490, 0.0024703, -0.0022424, 0.0027622, -0.0048113, 0.0047127
5: -0.0037235, 0.0067224, -0.0039368, 0.0071855, -0.0109090, 0.0106592
6: -0.0068225, -0.0009472, -0.0069413, -0.0006995, -0.0061230, 0.0059941
7: -0.0110654, 0.0004304, -0.0112918, 0.0006164, -0.0116818, 0.0117222
8: -0.0112907, 0.0001849, -0.0111212, 0.0006434, -0.0119341, 0.0113060
9: 0.9809808, 0.9978092, 0.9807566, 0.9986290, -0.0176482, 0.0170527

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125452, upper bound: 0.0111372
time: 1.44 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117130
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0056206, 0.0056552, -0.0108047, 0.0107087
1: 0.0022505, 0.0124151, 0.0017986, 0.0125274, -0.0102769, 0.0106165
2: 0.0142708, 0.0309785, 0.0139496, 0.0310403, -0.0144119, 0.0146107
3: -0.0097951, -0.0021241, -0.0099538, -0.0020377, -0.0077573, 0.0078297
4: -0.0022445, 0.0023594, -0.0022887, 0.0027693, -0.0050139, 0.0046481
5: -0.0034685, 0.0072581, -0.0039464, 0.0073299, -0.0107984, 0.0112045
6: -0.0066730, -0.0007098, -0.0069459, -0.0006251, -0.0060479, 0.0062361
7: -0.0111719, 0.0001152, -0.0114020, 0.0006266, -0.0117985, 0.0115172
8: -0.0109222, 0.0006468, -0.0111265, 0.0007786, -0.0117008, 0.0117733
9: 0.9808817, 0.9975876, 0.9806213, 0.9986537, -0.0177720, 0.0169663

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
time: 1.25 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0055071, 0.0056478, -0.0104269, 0.0111220
1: 0.0018756, 0.0118278, 0.0018118, 0.0123719, -0.0104963, 0.0100160
2: 0.0147073, 0.0317128, 0.0140726, 0.0310262, -0.0141858, 0.0152042
3: -0.0101313, -0.0025013, -0.0099500, -0.0021437, -0.0079877, 0.0074487
4: -0.0020544, 0.0025248, -0.0022424, 0.0027622, -0.0048166, 0.0047671
5: -0.0038068, 0.0067076, -0.0039368, 0.0071855, -0.0109923, 0.0106444
6: -0.0068669, -0.0009356, -0.0069413, -0.0006995, -0.0061674, 0.0060057
7: -0.0111134, 0.0004922, -0.0112918, 0.0006164, -0.0117298, 0.0117840
8: -0.0113331, 0.0001610, -0.0111212, 0.0006434, -0.0119765, 0.0112821
9: 0.9809233, 0.9979489, 0.9807566, 0.9986290, -0.0177057, 0.0171923

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123631, upper bound: 0.0111694
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0055529, 0.0054417, -0.0106647, 0.0107403
1: 0.0018636, 0.0125181, 0.0023100, 0.0124183, -0.0105547, 0.0102081
2: 0.0139272, 0.0309755, 0.0142510, 0.0309143, -0.0144639, 0.0141883
3: -0.0099271, -0.0020233, -0.0097684, -0.0021151, -0.0078120, 0.0077451
4: -0.0022731, 0.0027077, -0.0022332, 0.0023064, -0.0045795, 0.0049409
5: -0.0038639, 0.0073364, -0.0034008, 0.0072732, -0.0111372, 0.0107372
6: -0.0069024, -0.0006412, -0.0066324, -0.0007213, -0.0061811, 0.0059911
7: -0.0113537, 0.0005622, -0.0111300, 0.0000567, -0.0114104, 0.0116922
8: -0.0110881, 0.0008096, -0.0108863, 0.0006757, -0.0117639, 0.0116958
9: 0.9806883, 0.9985073, 0.9809389, 0.9974611, -0.0167727, 0.0175683

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111808, upper bound: 0.0127214
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131007
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055379, 0.0056024, -0.0051373, 0.0059815, -0.0110685, 0.0103595
1: 0.0018770, 0.0123614, 0.0019422, 0.0118258, -0.0099488, 0.0104193
2: 0.0140528, 0.0309615, 0.0146924, 0.0316475, -0.0150417, 0.0139677
3: -0.0099233, -0.0021293, -0.0101021, -0.0024892, -0.0074341, 0.0079728
4: -0.0022272, 0.0027005, -0.0020490, 0.0024703, -0.0046975, 0.0047495
5: -0.0038544, 0.0071915, -0.0037235, 0.0067224, -0.0105768, 0.0109149
6: -0.0068976, -0.0007165, -0.0068225, -0.0009472, -0.0059504, 0.0061060
7: -0.0112411, 0.0005519, -0.0110654, 0.0004304, -0.0116716, 0.0116173
8: -0.0110828, 0.0006735, -0.0112907, 0.0001849, -0.0112677, 0.0119642
9: 0.9808249, 0.9984818, 0.9809808, 0.9978092, -0.0169843, 0.0175011

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111377, upper bound: 0.0124086
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117141, upper bound: 0.0129544
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0055222, 0.0054834, -0.0107159, 0.0107285
1: 0.0018636, 0.0125181, 0.0022505, 0.0124151, -0.0105515, 0.0102676
2: 0.0139272, 0.0309755, 0.0142708, 0.0309785, -0.0145327, 0.0142068
3: -0.0099271, -0.0020233, -0.0097951, -0.0021241, -0.0078030, 0.0077717
4: -0.0022731, 0.0027077, -0.0022445, 0.0023594, -0.0046325, 0.0049522
5: -0.0038639, 0.0073364, -0.0034685, 0.0072581, -0.0111220, 0.0108050
6: -0.0069024, -0.0006412, -0.0066730, -0.0007098, -0.0061926, 0.0060317
7: -0.0113537, 0.0005622, -0.0111719, 0.0001152, -0.0114689, 0.0117341
8: -0.0110881, 0.0008096, -0.0109222, 0.0006468, -0.0117350, 0.0117318
9: 0.9806883, 0.9985073, 0.9808817, 0.9975876, -0.0168993, 0.0176256

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112146, upper bound: 0.0126763
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117537, upper bound: 0.0130438
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0055379, 0.0056024, -0.0051101, 0.0060261, -0.0111240, 0.0103514
1: 0.0018770, 0.0123614, 0.0018756, 0.0118278, -0.0099508, 0.0104858
2: 0.0140528, 0.0309615, 0.0147073, 0.0317128, -0.0151130, 0.0139868
3: -0.0099233, -0.0021293, -0.0101313, -0.0025013, -0.0074220, 0.0080020
4: -0.0022272, 0.0027005, -0.0020544, 0.0025248, -0.0047520, 0.0047549
5: -0.0038544, 0.0071915, -0.0038068, 0.0067076, -0.0105620, 0.0109983
6: -0.0068976, -0.0007165, -0.0068669, -0.0009356, -0.0059620, 0.0061504
7: -0.0112411, 0.0005519, -0.0111134, 0.0004922, -0.0117333, 0.0116653
8: -0.0110828, 0.0006735, -0.0113331, 0.0001610, -0.0112438, 0.0120065
9: 0.9808249, 0.9984818, 0.9809233, 0.9979489, -0.0171239, 0.0175585

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111694, upper bound: 0.0123607
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117474, upper bound: 0.0128989
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0055529, 0.0054417, -0.0106580, 0.0107942
1: 0.0017986, 0.0125278, 0.0023100, 0.0124183, -0.0106197, 0.0102178
2: 0.0139490, 0.0310403, 0.0142510, 0.0309143, -0.0144783, 0.0142585
3: -0.0099538, -0.0020374, -0.0097684, -0.0021151, -0.0078388, 0.0077310
4: -0.0022889, 0.0027694, -0.0022332, 0.0023064, -0.0045953, 0.0050025
5: -0.0039464, 0.0073304, -0.0034008, 0.0072732, -0.0112197, 0.0107312
6: -0.0069459, -0.0006249, -0.0066324, -0.0007213, -0.0062246, 0.0060075
7: -0.0114023, 0.0006266, -0.0111300, 0.0000567, -0.0114590, 0.0117567
8: -0.0111265, 0.0007791, -0.0108863, 0.0006757, -0.0118023, 0.0116654
9: 0.9806208, 0.9986536, 0.9809389, 0.9974611, -0.0168403, 0.0177147

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0128417
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117224, upper bound: 0.0131787
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055071, 0.0056478, -0.0051373, 0.0059815, -0.0110405, 0.0104133
1: 0.0018118, 0.0123719, 0.0019422, 0.0118258, -0.0100139, 0.0104298
2: 0.0140726, 0.0310262, 0.0146924, 0.0316475, -0.0149901, 0.0140382
3: -0.0099500, -0.0021437, -0.0101021, -0.0024892, -0.0074609, 0.0079585
4: -0.0022424, 0.0027622, -0.0020490, 0.0024703, -0.0047127, 0.0048113
5: -0.0039368, 0.0071855, -0.0037235, 0.0067224, -0.0106592, 0.0109090
6: -0.0069413, -0.0006995, -0.0068225, -0.0009472, -0.0059941, 0.0061230
7: -0.0112918, 0.0006164, -0.0110654, 0.0004304, -0.0117222, 0.0116818
8: -0.0111212, 0.0006434, -0.0112907, 0.0001849, -0.0113060, 0.0119341
9: 0.9807566, 0.9986290, 0.9809808, 0.9978092, -0.0170527, 0.0176482

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0125452
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117130, upper bound: 0.0130346
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0055222, 0.0054834, -0.0107283, 0.0108048
1: 0.0017986, 0.0125278, 0.0022505, 0.0124151, -0.0106166, 0.0102773
2: 0.0139490, 0.0310403, 0.0142708, 0.0309785, -0.0146456, 0.0144119
3: -0.0099538, -0.0020374, -0.0097951, -0.0021241, -0.0078297, 0.0077577
4: -0.0022889, 0.0027694, -0.0022445, 0.0023594, -0.0046483, 0.0050139
5: -0.0039464, 0.0073304, -0.0034685, 0.0072581, -0.0112046, 0.0107989
6: -0.0069459, -0.0006249, -0.0066730, -0.0007098, -0.0062361, 0.0060481
7: -0.0114023, 0.0006266, -0.0111719, 0.0001152, -0.0115174, 0.0117985
8: -0.0111265, 0.0007791, -0.0109222, 0.0006468, -0.0117733, 0.0117013
9: 0.9806208, 0.9986536, 0.9808817, 0.9975876, -0.0169668, 0.0177719

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111820, upper bound: 0.0128417
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131787
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0055071, 0.0056478, -0.0051101, 0.0060261, -0.0111220, 0.0104269
1: 0.0018118, 0.0123719, 0.0018756, 0.0118278, -0.0100160, 0.0104963
2: 0.0140726, 0.0310262, 0.0147073, 0.0317128, -0.0152042, 0.0141858
3: -0.0099500, -0.0021437, -0.0101313, -0.0025013, -0.0074487, 0.0079877
4: -0.0022424, 0.0027622, -0.0020544, 0.0025248, -0.0047671, 0.0048166
5: -0.0039368, 0.0071855, -0.0038068, 0.0067076, -0.0106444, 0.0109923
6: -0.0069413, -0.0006995, -0.0068669, -0.0009356, -0.0060057, 0.0061674
7: -0.0112918, 0.0006164, -0.0111134, 0.0004922, -0.0117840, 0.0117298
8: -0.0111212, 0.0006434, -0.0113331, 0.0001610, -0.0112821, 0.0119765
9: 0.9807566, 0.9986290, 0.9809233, 0.9979489, -0.0171923, 0.0177057

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111386, upper bound: 0.0125452
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117132, upper bound: 0.0130346
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054456, 0.0055944, -0.0056508, 0.0056098, -0.0107127, 0.0109196
1: 0.0018874, 0.0122853, 0.0018636, 0.0125181, -0.0106307, 0.0104217
2: 0.0141717, 0.0309453, 0.0139272, 0.0309755, -0.0144628, 0.0148185
3: -0.0099198, -0.0022108, -0.0099271, -0.0020233, -0.0078964, 0.0077163
4: -0.0022028, 0.0026934, -0.0022731, 0.0027077, -0.0049105, 0.0049665
5: -0.0038469, 0.0071203, -0.0038639, 0.0073364, -0.0111834, 0.0109843
6: -0.0068940, -0.0007561, -0.0069024, -0.0006412, -0.0062527, 0.0061462
7: -0.0111430, 0.0005440, -0.0113537, 0.0005622, -0.0117052, 0.0118977
8: -0.0110778, 0.0005692, -0.0110881, 0.0008096, -0.0118873, 0.0116574
9: 0.9809487, 0.9984610, 0.9806883, 0.9985073, -0.0175585, 0.0177727

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120697, upper bound: 0.0126777
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122316, upper bound: 0.0129858
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050327, 0.0061389, -0.0055379, 0.0056024, -0.0103239, 0.0113264
1: 0.0015071, 0.0117046, 0.0018770, 0.0123614, -0.0108543, 0.0098276
2: 0.0146140, 0.0316795, 0.0140528, 0.0309615, -0.0142443, 0.0153772
3: -0.0102562, -0.0025867, -0.0099233, -0.0021293, -0.0081270, 0.0073366
4: -0.0020237, 0.0028512, -0.0022272, 0.0027005, -0.0047242, 0.0050784
5: -0.0041739, 0.0065721, -0.0038544, 0.0071915, -0.0113654, 0.0104265
6: -0.0070874, -0.0009741, -0.0068976, -0.0007165, -0.0063709, 0.0059235
7: -0.0110848, 0.0009264, -0.0112411, 0.0005519, -0.0116367, 0.0121675
8: -0.0114869, 0.0000807, -0.0110828, 0.0006735, -0.0121604, 0.0111635
9: 0.9809882, 0.9987926, 0.9808249, 0.9984818, -0.0174936, 0.0179677

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118901, upper bound: 0.0126777
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121004, upper bound: 0.0129858
time: 2.12 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054456, 0.0055944, -0.0056210, 0.0056552, -0.0107670, 0.0109141
1: 0.0018874, 0.0122853, 0.0017986, 0.0125278, -0.0106404, 0.0104867
2: 0.0141717, 0.0309453, 0.0139490, 0.0310403, -0.0145329, 0.0148379
3: -0.0099198, -0.0022108, -0.0099538, -0.0020374, -0.0078824, 0.0077430
4: -0.0022028, 0.0026934, -0.0022889, 0.0027694, -0.0049721, 0.0049823
5: -0.0038469, 0.0071203, -0.0039464, 0.0073304, -0.0111773, 0.0110668
6: -0.0068940, -0.0007561, -0.0069459, -0.0006249, -0.0062691, 0.0061897
7: -0.0111430, 0.0005440, -0.0114023, 0.0006266, -0.0117696, 0.0119463
8: -0.0110778, 0.0005692, -0.0111265, 0.0007791, -0.0118569, 0.0116957
9: 0.9809487, 0.9984610, 0.9806208, 0.9986536, -0.0177048, 0.0178402

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121784, upper bound: 0.0125652
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122624, upper bound: 0.0129305
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050327, 0.0061389, -0.0055071, 0.0056478, -0.0103780, 0.0112869
1: 0.0015071, 0.0117046, 0.0018118, 0.0123719, -0.0108648, 0.0098927
2: 0.0146140, 0.0316795, 0.0140726, 0.0310262, -0.0143147, 0.0152318
3: -0.0102562, -0.0025867, -0.0099500, -0.0021437, -0.0081126, 0.0073633
4: -0.0020237, 0.0028512, -0.0022424, 0.0027622, -0.0047860, 0.0050935
5: -0.0041739, 0.0065721, -0.0039368, 0.0071855, -0.0113594, 0.0105089
6: -0.0070874, -0.0009741, -0.0069413, -0.0006995, -0.0063879, 0.0059672
7: -0.0110848, 0.0009264, -0.0112918, 0.0006164, -0.0117013, 0.0122182
8: -0.0114869, 0.0000807, -0.0111212, 0.0006434, -0.0121303, 0.0112019
9: 0.9809882, 0.9987926, 0.9807566, 0.9986290, -0.0176408, 0.0180361

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119464, upper bound: 0.0126583
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121436, upper bound: 0.0129305
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0054456, 0.0055944, -0.0109141, 0.0107670
1: 0.0017986, 0.0125278, 0.0018874, 0.0122853, -0.0104867, 0.0106404
2: 0.0139490, 0.0310403, 0.0141717, 0.0309453, -0.0148379, 0.0145329
3: -0.0099538, -0.0020374, -0.0099198, -0.0022108, -0.0077430, 0.0078824
4: -0.0022889, 0.0027694, -0.0022028, 0.0026934, -0.0049823, 0.0049721
5: -0.0039464, 0.0073304, -0.0038469, 0.0071203, -0.0110668, 0.0111773
6: -0.0069459, -0.0006249, -0.0068940, -0.0007561, -0.0061897, 0.0062691
7: -0.0114023, 0.0006266, -0.0111430, 0.0005440, -0.0119463, 0.0117696
8: -0.0111265, 0.0007791, -0.0110778, 0.0005692, -0.0116957, 0.0118569
9: 0.9806208, 0.9986536, 0.9809487, 0.9984610, -0.0178402, 0.0177048

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118951, upper bound: 0.0130875
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0131869
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055071, 0.0056478, -0.0050327, 0.0061389, -0.0112869, 0.0103780
1: 0.0018118, 0.0123719, 0.0015071, 0.0117046, -0.0098927, 0.0108648
2: 0.0140726, 0.0310262, 0.0146140, 0.0316795, -0.0152318, 0.0143147
3: -0.0099500, -0.0021437, -0.0102562, -0.0025867, -0.0073633, 0.0081126
4: -0.0022424, 0.0027622, -0.0020237, 0.0028512, -0.0050935, 0.0047860
5: -0.0039368, 0.0071855, -0.0041739, 0.0065721, -0.0105089, 0.0113594
6: -0.0069413, -0.0006995, -0.0070874, -0.0009741, -0.0059672, 0.0063879
7: -0.0112918, 0.0006164, -0.0110848, 0.0009264, -0.0122182, 0.0117013
8: -0.0111212, 0.0006434, -0.0114869, 0.0000807, -0.0112019, 0.0121303
9: 0.9807566, 0.9986290, 0.9809882, 0.9987926, -0.0180361, 0.0176408

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118609, upper bound: 0.0127118
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120965, upper bound: 0.0130680
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054195, 0.0056394, -0.0056210, 0.0056552, -0.0107860, 0.0109895
1: 0.0018224, 0.0122972, 0.0017986, 0.0125278, -0.0107053, 0.0104987
2: 0.0141922, 0.0310094, 0.0139490, 0.0310403, -0.0147036, 0.0150185
3: -0.0099463, -0.0022222, -0.0099538, -0.0020374, -0.0079089, 0.0077316
4: -0.0022188, 0.0027551, -0.0022889, 0.0027694, -0.0049881, 0.0050439
5: -0.0039291, 0.0071198, -0.0039464, 0.0073304, -0.0112595, 0.0110662
6: -0.0069377, -0.0007384, -0.0069459, -0.0006249, -0.0063128, 0.0062075
7: -0.0111910, 0.0006083, -0.0114023, 0.0006266, -0.0118176, 0.0120106
8: -0.0111158, 0.0005438, -0.0111265, 0.0007791, -0.0118949, 0.0116703
9: 0.9808795, 0.9986090, 0.9806208, 0.9986536, -0.0177741, 0.0179882

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050081, 0.0061870, -0.0055071, 0.0056478, -0.0103976, 0.0113711
1: 0.0014401, 0.0117137, 0.0018118, 0.0123719, -0.0109318, 0.0099019
2: 0.0146289, 0.0317444, 0.0140726, 0.0310262, -0.0144780, 0.0154516
3: -0.0102853, -0.0025958, -0.0099500, -0.0021437, -0.0081417, 0.0073542
4: -0.0020369, 0.0029138, -0.0022424, 0.0027622, -0.0047991, 0.0051562
5: -0.0042614, 0.0065708, -0.0039368, 0.0071855, -0.0114470, 0.0105076
6: -0.0071326, -0.0009575, -0.0069413, -0.0006995, -0.0064331, 0.0059838
7: -0.0111388, 0.0009869, -0.0112918, 0.0006164, -0.0117553, 0.0122787
8: -0.0115292, 0.0000599, -0.0111212, 0.0006434, -0.0121726, 0.0111811
9: 0.9809195, 0.9989452, 0.9807566, 0.9986290, -0.0177095, 0.0181887

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118904, upper bound: 0.0127790
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120965, upper bound: 0.0130680
time: 1.13 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.51 seconds
NS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117294, upper bound: 0.0117294
NS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117294, upper bound: 0.0117294
NS_A1_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0112211, upper bound: 0.0111377
NS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117230, upper bound: 0.0117230
NS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
NS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117272, upper bound: 0.0117597
NS_A1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0112609
NS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117210, upper bound: 0.0117562
NS_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117208
NS_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117208
NS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0111377
NS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117141
NS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
NS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0123607, upper bound: 0.0111694
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
NS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117597, upper bound: 0.0117272
NS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117597, upper bound: 0.0117272
NS_A1_B2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0112609, upper bound: 0.0111372
NS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117562, upper bound: 0.0117210
NS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117278, upper bound: 0.0117597
NS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117278, upper bound: 0.0117597
NS_A1_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0112204, upper bound: 0.0111694
NS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117215, upper bound: 0.0117562
NS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117195
NS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117195
NS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0125452, upper bound: 0.0111372
NS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117130
NS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
NS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117521
NS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0123631, upper bound: 0.0111694
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
NS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111808, upper bound: 0.0127214
NS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131007
NS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111377, upper bound: 0.0124086
NS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117141, upper bound: 0.0129544
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0112146, upper bound: 0.0126763
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117537, upper bound: 0.0130438
NS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111694, upper bound: 0.0123607
NS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117474, upper bound: 0.0128989
NS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0128417
NS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117224, upper bound: 0.0131787
NS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0125452
NS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117130, upper bound: 0.0130346
NS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111820, upper bound: 0.0128417
NS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131787
NS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0111386, upper bound: 0.0125452
NS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0117132, upper bound: 0.0130346
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0120697, upper bound: 0.0126777
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0122316, upper bound: 0.0129858
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0118901, upper bound: 0.0126777
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0121004, upper bound: 0.0129858
NS_A2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0121784, upper bound: 0.0125652
NS_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0122624, upper bound: 0.0129305
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0119464, upper bound: 0.0126583
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0121436, upper bound: 0.0129305
NS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0118951, upper bound: 0.0130875
NS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0131869
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0118609, upper bound: 0.0127118
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0120965, upper bound: 0.0130680
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0121017, upper bound: 0.0130680
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0118904, upper bound: 0.0127790
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 9, lower bound: -0.0120965, upper bound: 0.0130680

## BFS NS instance: NS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0055529, 0.0054417, -0.0105218, 0.0105218
1: 0.0023100, 0.0124183, 0.0023100, 0.0124183, -0.0101082, 0.0101082
2: 0.0142510, 0.0309143, 0.0142510, 0.0309143, -0.0141019, 0.0141019
3: -0.0097684, -0.0021151, -0.0097684, -0.0021151, -0.0076533, 0.0076533
4: -0.0022332, 0.0023064, -0.0022332, 0.0023064, -0.0045396, 0.0045396
5: -0.0034008, 0.0072732, -0.0034008, 0.0072732, -0.0106740, 0.0106740
6: -0.0066324, -0.0007213, -0.0066324, -0.0007213, -0.0059111, 0.0059111
7: -0.0111300, 0.0000567, -0.0111300, 0.0000567, -0.0111867, 0.0111867
8: -0.0108863, 0.0006757, -0.0108863, 0.0006757, -0.0115620, 0.0115620
9: 0.9809389, 0.9974611, 0.9809389, 0.9974611, -0.0165222, 0.0165222

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118339, upper bound: 0.0114395
time: 1.04 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120331, upper bound: 0.0117351
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0051373, 0.0059815, -0.0110474, 0.0100945
1: 0.0023100, 0.0124183, 0.0019422, 0.0118258, -0.0095157, 0.0104761
2: 0.0142510, 0.0309143, 0.0146924, 0.0316475, -0.0148240, 0.0136293
3: -0.0097684, -0.0021151, -0.0101021, -0.0024892, -0.0072793, 0.0079870
4: -0.0022332, 0.0023064, -0.0020490, 0.0024703, -0.0047035, 0.0043555
5: -0.0034008, 0.0072732, -0.0037235, 0.0067224, -0.0101232, 0.0109967
6: -0.0066324, -0.0007213, -0.0068225, -0.0009472, -0.0056852, 0.0061012
7: -0.0111300, 0.0000567, -0.0110654, 0.0004304, -0.0115605, 0.0111221
8: -0.0108863, 0.0006757, -0.0112907, 0.0001849, -0.0110711, 0.0119665
9: 0.9809389, 0.9974611, 0.9809808, 0.9978092, -0.0168703, 0.0164803

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118339, upper bound: 0.0114395
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120331, upper bound: 0.0117351
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0056083, 0.0054130, -0.0101254, 0.0110911
1: 0.0019422, 0.0118258, 0.0023241, 0.0124157, -0.0104735, 0.0095016
2: 0.0146924, 0.0316475, 0.0141685, 0.0308714, -0.0138352, 0.0148445
3: -0.0101021, -0.0024892, -0.0097530, -0.0020835, -0.0080187, 0.0072638
4: -0.0020490, 0.0024703, -0.0022497, 0.0023008, -0.0043498, 0.0047200
5: -0.0037235, 0.0067224, -0.0033860, 0.0073167, -0.0110402, 0.0101084
6: -0.0068225, -0.0009472, -0.0066242, -0.0007068, -0.0061157, 0.0056770
7: -0.0110654, 0.0004304, -0.0111395, 0.0000431, -0.0111085, 0.0115700
8: -0.0112907, 0.0001849, -0.0108646, 0.0007525, -0.0120432, 0.0110494
9: 0.9809808, 0.9978092, 0.9808846, 0.9974533, -0.0164726, 0.0169246

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117230, upper bound: 0.0117230
time: 1.60 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117230, upper bound: 0.0117230
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0055529, 0.0054417, -0.0105100, 0.0105722
1: 0.0022505, 0.0124151, 0.0023100, 0.0124183, -0.0101678, 0.0101051
2: 0.0142708, 0.0309785, 0.0142510, 0.0309143, -0.0141204, 0.0141718
3: -0.0097951, -0.0021241, -0.0097684, -0.0021151, -0.0076800, 0.0076443
4: -0.0022445, 0.0023594, -0.0022332, 0.0023064, -0.0045510, 0.0045926
5: -0.0034685, 0.0072581, -0.0034008, 0.0072732, -0.0107418, 0.0106589
6: -0.0066730, -0.0007098, -0.0066324, -0.0007213, -0.0059517, 0.0059226
7: -0.0111719, 0.0001152, -0.0111300, 0.0000567, -0.0112286, 0.0112452
8: -0.0109222, 0.0006468, -0.0108863, 0.0006757, -0.0115980, 0.0115331
9: 0.9808817, 0.9975876, 0.9809389, 0.9974611, -0.0165794, 0.0166487

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114340, upper bound: 0.0118656
time: 1.06 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117339, upper bound: 0.0120764
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0055529, 0.0054417, -0.0100848, 0.0111031
1: 0.0018756, 0.0118278, 0.0023100, 0.0124183, -0.0105426, 0.0095178
2: 0.0147073, 0.0317128, 0.0142510, 0.0309143, -0.0136474, 0.0148960
3: -0.0101313, -0.0025013, -0.0097684, -0.0021151, -0.0080162, 0.0072671
4: -0.0020544, 0.0025248, -0.0022332, 0.0023064, -0.0043608, 0.0047580
5: -0.0038068, 0.0067076, -0.0034008, 0.0072732, -0.0110800, 0.0101084
6: -0.0068669, -0.0009356, -0.0066324, -0.0007213, -0.0061457, 0.0056968
7: -0.0111134, 0.0004922, -0.0111300, 0.0000567, -0.0111701, 0.0116223
8: -0.0113331, 0.0001610, -0.0108863, 0.0006757, -0.0120088, 0.0110472
9: 0.9809233, 0.9979489, 0.9809389, 0.9974611, -0.0165378, 0.0170100

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114340, upper bound: 0.0118656
time: 1.54 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117339, upper bound: 0.0120764
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055761, 0.0054548, -0.0051373, 0.0059815, -0.0110817, 0.0101760
1: 0.0022650, 0.0124160, 0.0019422, 0.0118258, -0.0095608, 0.0104739
2: 0.0141865, 0.0309358, 0.0146924, 0.0316475, -0.0148636, 0.0139046
3: -0.0097797, -0.0020942, -0.0101021, -0.0024892, -0.0072905, 0.0080079
4: -0.0022616, 0.0023535, -0.0020490, 0.0024703, -0.0047319, 0.0044026
5: -0.0034536, 0.0073046, -0.0037235, 0.0067224, -0.0101760, 0.0110280
6: -0.0066646, -0.0006967, -0.0068225, -0.0009472, -0.0057174, 0.0061258
7: -0.0111850, 0.0001014, -0.0110654, 0.0004304, -0.0116154, 0.0111668
8: -0.0109007, 0.0007202, -0.0112907, 0.0001849, -0.0110855, 0.0120110
9: 0.9808258, 0.9975786, 0.9809808, 0.9978092, -0.0169834, 0.0165978

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117210, upper bound: 0.0117562
time: 1.01 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117210, upper bound: 0.0117562
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0054456, 0.0055944, -0.0107186, 0.0104495
1: 0.0023100, 0.0124183, 0.0018874, 0.0122853, -0.0099752, 0.0105309
2: 0.0142510, 0.0309143, 0.0141717, 0.0309453, -0.0141396, 0.0141637
3: -0.0097684, -0.0021151, -0.0099198, -0.0022108, -0.0075576, 0.0078047
4: -0.0022332, 0.0023064, -0.0022028, 0.0026934, -0.0049266, 0.0045092
5: -0.0034008, 0.0072732, -0.0038469, 0.0071203, -0.0105211, 0.0111202
6: -0.0066324, -0.0007213, -0.0068940, -0.0007561, -0.0058762, 0.0061727
7: -0.0111300, 0.0000567, -0.0111430, 0.0005440, -0.0116740, 0.0111997
8: -0.0108863, 0.0006757, -0.0110778, 0.0005692, -0.0114555, 0.0117535
9: 0.9809389, 0.9974611, 0.9809487, 0.9984610, -0.0175221, 0.0165123

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127214, upper bound: 0.0111808
time: 1.12 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131007, upper bound: 0.0117227
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0050327, 0.0061389, -0.0112481, 0.0100147
1: 0.0023100, 0.0124183, 0.0015071, 0.0117046, -0.0093945, 0.0109111
2: 0.0142510, 0.0309143, 0.0146140, 0.0316795, -0.0148693, 0.0136899
3: -0.0097684, -0.0021151, -0.0102562, -0.0025867, -0.0071817, 0.0081412
4: -0.0022332, 0.0023064, -0.0020237, 0.0028512, -0.0050843, 0.0043302
5: -0.0034008, 0.0072732, -0.0041739, 0.0065721, -0.0099729, 0.0114471
6: -0.0066324, -0.0007213, -0.0070874, -0.0009741, -0.0056583, 0.0063661
7: -0.0111300, 0.0000567, -0.0110848, 0.0009264, -0.0120564, 0.0111415
8: -0.0108863, 0.0006757, -0.0114869, 0.0000807, -0.0109669, 0.0121627
9: 0.9809389, 0.9974611, 0.9809882, 0.9987926, -0.0178537, 0.0164729

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129654, upper bound: 0.0114347
time: 1.08 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131007, upper bound: 0.0117227
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0051233, 0.0059621, -0.0054593, 0.0055861, -0.0103285, 0.0109551
1: 0.0019534, 0.0117993, 0.0018457, 0.0122283, -0.0102749, 0.0099535
2: 0.0147073, 0.0316157, 0.0141371, 0.0308985, -0.0138407, 0.0147938
3: -0.0100924, -0.0025114, -0.0099402, -0.0022487, -0.0078437, 0.0074288
4: -0.0020468, 0.0024642, -0.0022165, 0.0026935, -0.0047402, 0.0046807
5: -0.0037130, 0.0067135, -0.0038299, 0.0071502, -0.0108632, 0.0105435
6: -0.0068169, -0.0009576, -0.0069032, -0.0007610, -0.0060559, 0.0059456
7: -0.0110273, 0.0004206, -0.0110932, 0.0006184, -0.0116457, 0.0115138
8: -0.0112769, 0.0001720, -0.0110528, 0.0006045, -0.0118814, 0.0112249
9: 0.9810164, 0.9977967, 0.9809167, 0.9984845, -0.0174681, 0.0168800

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0111377
time: 1.05 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0111377
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0055008, 0.0055654, -0.0103213, 0.0110137
1: 0.0019422, 0.0118258, 0.0019014, 0.0122909, -0.0103487, 0.0099243
2: 0.0146924, 0.0316475, 0.0140902, 0.0309018, -0.0138728, 0.0149069
3: -0.0101021, -0.0024892, -0.0099040, -0.0021854, -0.0079167, 0.0074148
4: -0.0020490, 0.0024703, -0.0022233, 0.0026880, -0.0047370, 0.0046936
5: -0.0037235, 0.0067224, -0.0038318, 0.0071722, -0.0108957, 0.0105542
6: -0.0068225, -0.0009472, -0.0068858, -0.0007401, -0.0060824, 0.0059387
7: -0.0110654, 0.0004304, -0.0111528, 0.0005308, -0.0115962, 0.0115832
8: -0.0112907, 0.0001849, -0.0110558, 0.0006410, -0.0119317, 0.0112407
9: 0.9809808, 0.9978092, 0.9808924, 0.9984529, -0.0174721, 0.0169169

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117141
time: 1.24 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117141
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0054456, 0.0055944, -0.0107068, 0.0105000
1: 0.0022505, 0.0124151, 0.0018874, 0.0122853, -0.0100348, 0.0105277
2: 0.0142708, 0.0309785, 0.0141717, 0.0309453, -0.0141581, 0.0142336
3: -0.0097951, -0.0021241, -0.0099198, -0.0022108, -0.0075843, 0.0077956
4: -0.0022445, 0.0023594, -0.0022028, 0.0026934, -0.0049380, 0.0045622
5: -0.0034685, 0.0072581, -0.0038469, 0.0071203, -0.0105889, 0.0111050
6: -0.0066730, -0.0007098, -0.0068940, -0.0007561, -0.0059168, 0.0061842
7: -0.0111719, 0.0001152, -0.0111430, 0.0005440, -0.0117159, 0.0112582
8: -0.0109222, 0.0006468, -0.0110778, 0.0005692, -0.0114914, 0.0117246
9: 0.9808817, 0.9975876, 0.9809487, 0.9984610, -0.0175793, 0.0166389

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126763, upper bound: 0.0112146
time: 1.19 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0050327, 0.0061389, -0.0112363, 0.0100651
1: 0.0022505, 0.0124151, 0.0015071, 0.0117046, -0.0094541, 0.0109080
2: 0.0142708, 0.0309785, 0.0146140, 0.0316795, -0.0148878, 0.0137597
3: -0.0097951, -0.0021241, -0.0102562, -0.0025867, -0.0072083, 0.0081321
4: -0.0022445, 0.0023594, -0.0020237, 0.0028512, -0.0050957, 0.0043831
5: -0.0034685, 0.0072581, -0.0041739, 0.0065721, -0.0100407, 0.0114320
6: -0.0066730, -0.0007098, -0.0070874, -0.0009741, -0.0056989, 0.0063776
7: -0.0111719, 0.0001152, -0.0110848, 0.0009264, -0.0120983, 0.0112000
8: -0.0109222, 0.0006468, -0.0114869, 0.0000807, -0.0110029, 0.0121338
9: 0.9808817, 0.9975876, 0.9809882, 0.9987926, -0.0179110, 0.0165994

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129288, upper bound: 0.0114822
time: 1.08 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0050960, 0.0060071, -0.0054593, 0.0055861, -0.0103204, 0.0110112
1: 0.0018868, 0.0118011, 0.0018457, 0.0122283, -0.0103415, 0.0099554
2: 0.0147222, 0.0316817, 0.0141371, 0.0308985, -0.0138598, 0.0148663
3: -0.0101218, -0.0025238, -0.0099402, -0.0022487, -0.0078731, 0.0074164
4: -0.0020521, 0.0025187, -0.0022165, 0.0026935, -0.0047456, 0.0047351
5: -0.0037964, 0.0066987, -0.0038299, 0.0071502, -0.0109466, 0.0105286
6: -0.0068614, -0.0009460, -0.0069032, -0.0007610, -0.0061004, 0.0059572
7: -0.0110757, 0.0004826, -0.0110932, 0.0006184, -0.0116941, 0.0115757
8: -0.0113195, 0.0001481, -0.0110528, 0.0006045, -0.0119240, 0.0112010
9: 0.9809585, 0.9979365, 0.9809167, 0.9984845, -0.0175260, 0.0170198

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123607, upper bound: 0.0111694
time: 1.03 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123607, upper bound: 0.0111694
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0055008, 0.0055654, -0.0103133, 0.0110685
1: 0.0018756, 0.0118278, 0.0019014, 0.0122909, -0.0104152, 0.0099263
2: 0.0147073, 0.0317128, 0.0140902, 0.0309018, -0.0138918, 0.0149784
3: -0.0101313, -0.0025013, -0.0099040, -0.0021854, -0.0079459, 0.0074026
4: -0.0020544, 0.0025248, -0.0022233, 0.0026880, -0.0047424, 0.0047481
5: -0.0038068, 0.0067076, -0.0038318, 0.0071722, -0.0109790, 0.0105394
6: -0.0068669, -0.0009356, -0.0068858, -0.0007401, -0.0061268, 0.0059502
7: -0.0111134, 0.0004922, -0.0111528, 0.0005308, -0.0116443, 0.0116450
8: -0.0113331, 0.0001610, -0.0110558, 0.0006410, -0.0119741, 0.0112168
9: 0.9809233, 0.9979489, 0.9808924, 0.9984529, -0.0175296, 0.0170565

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
time: 1.84 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0055222, 0.0054834, -0.0105722, 0.0105100
1: 0.0023100, 0.0124183, 0.0022505, 0.0124151, -0.0101051, 0.0101678
2: 0.0142510, 0.0309143, 0.0142708, 0.0309785, -0.0141718, 0.0141204
3: -0.0097684, -0.0021151, -0.0097951, -0.0021241, -0.0076443, 0.0076800
4: -0.0022332, 0.0023064, -0.0022445, 0.0023594, -0.0045926, 0.0045510
5: -0.0034008, 0.0072732, -0.0034685, 0.0072581, -0.0106589, 0.0107418
6: -0.0066324, -0.0007213, -0.0066730, -0.0007098, -0.0059226, 0.0059517
7: -0.0111300, 0.0000567, -0.0111719, 0.0001152, -0.0112452, 0.0112286
8: -0.0108863, 0.0006757, -0.0109222, 0.0006468, -0.0115331, 0.0115980
9: 0.9809389, 0.9974611, 0.9808817, 0.9975876, -0.0166487, 0.0165794

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118656, upper bound: 0.0114340
time: 1.07 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120764, upper bound: 0.0117339
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0051101, 0.0060261, -0.0111030, 0.0100848
1: 0.0023100, 0.0124183, 0.0018756, 0.0118278, -0.0095178, 0.0105426
2: 0.0142510, 0.0309143, 0.0147073, 0.0317128, -0.0148960, 0.0136474
3: -0.0097684, -0.0021151, -0.0101313, -0.0025013, -0.0072671, 0.0080162
4: -0.0022332, 0.0023064, -0.0020544, 0.0025248, -0.0047580, 0.0043608
5: -0.0034008, 0.0072732, -0.0038068, 0.0067076, -0.0101084, 0.0110800
6: -0.0066324, -0.0007213, -0.0068669, -0.0009356, -0.0056968, 0.0061457
7: -0.0111300, 0.0000567, -0.0111134, 0.0004922, -0.0116223, 0.0111701
8: -0.0108863, 0.0006757, -0.0113331, 0.0001610, -0.0110472, 0.0120088
9: 0.9809389, 0.9974611, 0.9809233, 0.9979489, -0.0170100, 0.0165378

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118656, upper bound: 0.0114340
time: 1.09 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120764, upper bound: 0.0117339
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0055761, 0.0054548, -0.0101760, 0.0110817
1: 0.0019422, 0.0118258, 0.0022650, 0.0124160, -0.0104739, 0.0095608
2: 0.0146924, 0.0316475, 0.0141865, 0.0309358, -0.0139046, 0.0148636
3: -0.0101021, -0.0024892, -0.0097797, -0.0020942, -0.0080079, 0.0072905
4: -0.0020490, 0.0024703, -0.0022616, 0.0023535, -0.0044026, 0.0047319
5: -0.0037235, 0.0067224, -0.0034536, 0.0073046, -0.0110280, 0.0101760
6: -0.0068225, -0.0009472, -0.0066646, -0.0006967, -0.0061258, 0.0057174
7: -0.0110654, 0.0004304, -0.0111850, 0.0001014, -0.0111668, 0.0116154
8: -0.0112907, 0.0001849, -0.0109007, 0.0007202, -0.0120110, 0.0110855
9: 0.9809808, 0.9978092, 0.9808258, 0.9975786, -0.0165978, 0.0169834

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117562, upper bound: 0.0117210
time: 1.01 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117562, upper bound: 0.0117210
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0055222, 0.0054834, -0.0105821, 0.0105821
1: 0.0022505, 0.0124151, 0.0022505, 0.0124151, -0.0101647, 0.0101647
2: 0.0142708, 0.0309785, 0.0142708, 0.0309785, -0.0143243, 0.0143243
3: -0.0097951, -0.0021241, -0.0097951, -0.0021241, -0.0076709, 0.0076709
4: -0.0022445, 0.0023594, -0.0022445, 0.0023594, -0.0046040, 0.0046040
5: -0.0034685, 0.0072581, -0.0034685, 0.0072581, -0.0107267, 0.0107267
6: -0.0066730, -0.0007098, -0.0066730, -0.0007098, -0.0059632, 0.0059632
7: -0.0111719, 0.0001152, -0.0111719, 0.0001152, -0.0112871, 0.0112871
8: -0.0109222, 0.0006468, -0.0109222, 0.0006468, -0.0115690, 0.0115690
9: 0.9808817, 0.9975876, 0.9808817, 0.9975876, -0.0167059, 0.0167059

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118371, upper bound: 0.0114887
time: 1.74 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120315, upper bound: 0.0117643
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0051101, 0.0060261, -0.0111154, 0.0101564
1: 0.0022505, 0.0124151, 0.0018756, 0.0118278, -0.0095773, 0.0105395
2: 0.0142708, 0.0309785, 0.0147073, 0.0317128, -0.0150551, 0.0138504
3: -0.0097951, -0.0021241, -0.0101313, -0.0025013, -0.0072937, 0.0080072
4: -0.0022445, 0.0023594, -0.0020544, 0.0025248, -0.0047693, 0.0044138
5: -0.0034685, 0.0072581, -0.0038068, 0.0067076, -0.0101761, 0.0110649
6: -0.0066730, -0.0007098, -0.0068669, -0.0009356, -0.0057374, 0.0061571
7: -0.0111719, 0.0001152, -0.0111134, 0.0004922, -0.0116641, 0.0112286
8: -0.0109222, 0.0006468, -0.0113331, 0.0001610, -0.0110832, 0.0119799
9: 0.9808817, 0.9975876, 0.9809233, 0.9979489, -0.0170672, 0.0166643

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118371, upper bound: 0.0114887
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120315, upper bound: 0.0117643
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0055761, 0.0054548, -0.0101891, 0.0111603
1: 0.0018756, 0.0118278, 0.0022650, 0.0124160, -0.0105404, 0.0095628
2: 0.0147073, 0.0317128, 0.0141865, 0.0309358, -0.0140526, 0.0150694
3: -0.0101313, -0.0025013, -0.0097797, -0.0020942, -0.0080371, 0.0072784
4: -0.0020544, 0.0025248, -0.0022616, 0.0023535, -0.0044079, 0.0047864
5: -0.0038068, 0.0067076, -0.0034536, 0.0073046, -0.0111114, 0.0101612
6: -0.0068669, -0.0009356, -0.0066646, -0.0006967, -0.0061703, 0.0057290
7: -0.0111134, 0.0004922, -0.0111850, 0.0001014, -0.0112148, 0.0116772
8: -0.0113331, 0.0001610, -0.0109007, 0.0007202, -0.0120533, 0.0110616
9: 0.9809233, 0.9979489, 0.9808258, 0.9975786, -0.0166553, 0.0171230

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117215, upper bound: 0.0117562
time: 1.11 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117215, upper bound: 0.0117562
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0054195, 0.0056394, -0.0107726, 0.0104429
1: 0.0023100, 0.0124183, 0.0018224, 0.0122972, -0.0099872, 0.0105958
2: 0.0142510, 0.0309143, 0.0141922, 0.0310094, -0.0142101, 0.0141834
3: -0.0097684, -0.0021151, -0.0099463, -0.0022222, -0.0075462, 0.0078312
4: -0.0022332, 0.0023064, -0.0022188, 0.0027551, -0.0049883, 0.0045252
5: -0.0034008, 0.0072732, -0.0039291, 0.0071198, -0.0105206, 0.0112023
6: -0.0066324, -0.0007213, -0.0069377, -0.0007384, -0.0058940, 0.0062164
7: -0.0111300, 0.0000567, -0.0111910, 0.0006083, -0.0117384, 0.0112477
8: -0.0108863, 0.0006757, -0.0111158, 0.0005438, -0.0114300, 0.0117916
9: 0.9809389, 0.9974611, 0.9808795, 0.9986090, -0.0176701, 0.0165816

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128417, upper bound: 0.0111805
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131787, upper bound: 0.0117224
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0050081, 0.0061870, -0.0113053, 0.0100104
1: 0.0023100, 0.0124183, 0.0014401, 0.0117137, -0.0094037, 0.0109782
2: 0.0142510, 0.0309143, 0.0146289, 0.0317444, -0.0149410, 0.0137088
3: -0.0097684, -0.0021151, -0.0102853, -0.0025958, -0.0071726, 0.0081702
4: -0.0022332, 0.0023064, -0.0020369, 0.0029138, -0.0051470, 0.0043433
5: -0.0034008, 0.0072732, -0.0042614, 0.0065708, -0.0099716, 0.0115347
6: -0.0066324, -0.0007213, -0.0071326, -0.0009575, -0.0056749, 0.0064113
7: -0.0111300, 0.0000567, -0.0111388, 0.0009869, -0.0121170, 0.0111955
8: -0.0108863, 0.0006757, -0.0115292, 0.0000599, -0.0109461, 0.0122050
9: 0.9809389, 0.9974611, 0.9809195, 0.9989452, -0.0180063, 0.0165416

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130589, upper bound: 0.0114297
time: 1.26 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131787, upper bound: 0.0117224
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0051233, 0.0059621, -0.0054297, 0.0056335, -0.0103856, 0.0109503
1: 0.0019534, 0.0117993, 0.0017844, 0.0122391, -0.0102857, 0.0100149
2: 0.0147073, 0.0316157, 0.0141554, 0.0309691, -0.0139163, 0.0148146
3: -0.0100924, -0.0025114, -0.0099673, -0.0022608, -0.0078315, 0.0074559
4: -0.0020468, 0.0024642, -0.0022321, 0.0027554, -0.0048022, 0.0046963
5: -0.0037130, 0.0067135, -0.0039121, 0.0071459, -0.0108589, 0.0106257
6: -0.0068169, -0.0009576, -0.0069455, -0.0007426, -0.0060744, 0.0059878
7: -0.0110273, 0.0004206, -0.0111453, 0.0006781, -0.0117055, 0.0115659
8: -0.0112769, 0.0001720, -0.0110932, 0.0005761, -0.0118530, 0.0112652
9: 0.9810164, 0.9977967, 0.9808418, 0.9986300, -0.0176136, 0.0169548

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125452, upper bound: 0.0111372
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125452, upper bound: 0.0111372
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0054714, 0.0056101, -0.0103747, 0.0110089
1: 0.0019422, 0.0118258, 0.0018372, 0.0123032, -0.0103610, 0.0099886
2: 0.0146924, 0.0316475, 0.0141084, 0.0309659, -0.0139424, 0.0149254
3: -0.0101021, -0.0024892, -0.0099304, -0.0021984, -0.0079037, 0.0074412
4: -0.0020490, 0.0024703, -0.0022387, 0.0027493, -0.0047983, 0.0047090
5: -0.0037235, 0.0067224, -0.0039135, 0.0071675, -0.0108909, 0.0106360
6: -0.0068225, -0.0009472, -0.0069291, -0.0007225, -0.0061000, 0.0059820
7: -0.0110654, 0.0004304, -0.0112057, 0.0005947, -0.0116601, 0.0116362
8: -0.0112907, 0.0001849, -0.0110937, 0.0006124, -0.0119031, 0.0112786
9: 0.9809808, 0.9978092, 0.9808234, 0.9985991, -0.0176184, 0.0169858

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117130
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117130
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0054195, 0.0056394, -0.0107829, 0.0105155
1: 0.0022505, 0.0124151, 0.0018224, 0.0122972, -0.0100468, 0.0105927
2: 0.0142708, 0.0309785, 0.0141922, 0.0310094, -0.0143619, 0.0143869
3: -0.0097951, -0.0021241, -0.0099463, -0.0022222, -0.0075728, 0.0078222
4: -0.0022445, 0.0023594, -0.0022188, 0.0027551, -0.0049996, 0.0045782
5: -0.0034685, 0.0072581, -0.0039291, 0.0071198, -0.0105883, 0.0111872
6: -0.0066730, -0.0007098, -0.0069377, -0.0007384, -0.0059346, 0.0062279
7: -0.0111719, 0.0001152, -0.0111910, 0.0006083, -0.0117802, 0.0113061
8: -0.0109222, 0.0006468, -0.0111158, 0.0005438, -0.0114660, 0.0117627
9: 0.9808817, 0.9975876, 0.9808795, 0.9986090, -0.0177273, 0.0167081

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126830, upper bound: 0.0112146
time: 1.08 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0050081, 0.0061870, -0.0113177, 0.0100829
1: 0.0022505, 0.0124151, 0.0014401, 0.0117137, -0.0094632, 0.0109750
2: 0.0142708, 0.0309785, 0.0146289, 0.0317444, -0.0150988, 0.0139106
3: -0.0097951, -0.0021241, -0.0102853, -0.0025958, -0.0071992, 0.0081612
4: -0.0022445, 0.0023594, -0.0020369, 0.0029138, -0.0051584, 0.0043963
5: -0.0034685, 0.0072581, -0.0042614, 0.0065708, -0.0100393, 0.0115196
6: -0.0066730, -0.0007098, -0.0071326, -0.0009575, -0.0057154, 0.0064228
7: -0.0111719, 0.0001152, -0.0111388, 0.0009869, -0.0121588, 0.0112540
8: -0.0109222, 0.0006468, -0.0115292, 0.0000599, -0.0109821, 0.0121761
9: 0.9808817, 0.9975876, 0.9809195, 0.9989452, -0.0180635, 0.0166681

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129302, upper bound: 0.0114822
time: 1.31 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0050960, 0.0060071, -0.0054297, 0.0056335, -0.0103985, 0.0110317
1: 0.0018868, 0.0118011, 0.0017844, 0.0122391, -0.0103523, 0.0100167
2: 0.0147222, 0.0316817, 0.0141554, 0.0309691, -0.0140573, 0.0150295
3: -0.0101218, -0.0025238, -0.0099673, -0.0022608, -0.0078610, 0.0074435
4: -0.0020521, 0.0025187, -0.0022321, 0.0027554, -0.0048075, 0.0047507
5: -0.0037964, 0.0066987, -0.0039121, 0.0071459, -0.0109423, 0.0106108
6: -0.0068614, -0.0009460, -0.0069455, -0.0007426, -0.0061188, 0.0059994
7: -0.0110757, 0.0004826, -0.0111453, 0.0006781, -0.0117538, 0.0116279
8: -0.0113195, 0.0001481, -0.0110932, 0.0005761, -0.0118956, 0.0112413
9: 0.9809585, 0.9979365, 0.9808418, 0.9986300, -0.0176715, 0.0170947

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123631, upper bound: 0.0111694
time: 1.15 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123631, upper bound: 0.0111694
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0054714, 0.0056101, -0.0103883, 0.0110889
1: 0.0018756, 0.0118278, 0.0018372, 0.0123032, -0.0104276, 0.0099906
2: 0.0147073, 0.0317128, 0.0141084, 0.0309659, -0.0140893, 0.0151305
3: -0.0101313, -0.0025013, -0.0099304, -0.0021984, -0.0079329, 0.0074291
4: -0.0020544, 0.0025248, -0.0022387, 0.0027493, -0.0048037, 0.0047635
5: -0.0038068, 0.0067076, -0.0039135, 0.0071675, -0.0109743, 0.0106211
6: -0.0068669, -0.0009356, -0.0069291, -0.0007225, -0.0061444, 0.0059935
7: -0.0111134, 0.0004922, -0.0112057, 0.0005947, -0.0117081, 0.0116980
8: -0.0113331, 0.0001610, -0.0110937, 0.0006124, -0.0119455, 0.0112547
9: 0.9809233, 0.9979489, 0.9808234, 0.9985991, -0.0176758, 0.0171255

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
time: 1.11 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055729, 0.0055935, -0.0055394, 0.0054222, -0.0105512, 0.0107102
1: 0.0018327, 0.0123846, 0.0023217, 0.0123942, -0.0105615, 0.0100628
2: 0.0140116, 0.0309130, 0.0142653, 0.0308822, -0.0142082, 0.0140626
3: -0.0099440, -0.0021430, -0.0097585, -0.0021359, -0.0078081, 0.0076155
4: -0.0022622, 0.0027006, -0.0022310, 0.0023002, -0.0045624, 0.0049316
5: -0.0038389, 0.0072948, -0.0033898, 0.0072652, -0.0111041, 0.0106846
6: -0.0069078, -0.0006861, -0.0066267, -0.0007314, -0.0061763, 0.0059406
7: -0.0112056, 0.0006285, -0.0110926, 0.0000465, -0.0112522, 0.0117211
8: -0.0110580, 0.0007402, -0.0108721, 0.0006634, -0.0117214, 0.0116123
9: 0.9807805, 0.9985095, 0.9809726, 0.9974478, -0.0166673, 0.0175368

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111808, upper bound: 0.0127214
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111808, upper bound: 0.0127214
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056130, 0.0055728, -0.0055529, 0.0054417, -0.0106387, 0.0107020
1: 0.0018883, 0.0124472, 0.0023100, 0.0124183, -0.0105300, 0.0101371
2: 0.0139657, 0.0309155, 0.0142510, 0.0309143, -0.0144360, 0.0140921
3: -0.0099077, -0.0020804, -0.0097684, -0.0021151, -0.0077927, 0.0076880
4: -0.0022691, 0.0026951, -0.0022332, 0.0023064, -0.0045755, 0.0049283
5: -0.0038412, 0.0073168, -0.0034008, 0.0072732, -0.0111144, 0.0107176
6: -0.0068905, -0.0006653, -0.0066324, -0.0007213, -0.0061693, 0.0059671
7: -0.0112642, 0.0005409, -0.0111300, 0.0000567, -0.0113209, 0.0116710
8: -0.0110610, 0.0007766, -0.0108863, 0.0006757, -0.0117367, 0.0116628
9: 0.9807566, 0.9984781, 0.9809389, 0.9974611, -0.0167045, 0.0175391

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131007
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131007
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054593, 0.0055861, -0.0051233, 0.0059621, -0.0109551, 0.0103285
1: 0.0018457, 0.0122283, 0.0019534, 0.0117993, -0.0099535, 0.0102749
2: 0.0141371, 0.0308985, 0.0147073, 0.0316157, -0.0147938, 0.0138407
3: -0.0099402, -0.0022487, -0.0100924, -0.0025114, -0.0074288, 0.0078437
4: -0.0022165, 0.0026935, -0.0020468, 0.0024642, -0.0046807, 0.0047402
5: -0.0038299, 0.0071502, -0.0037130, 0.0067135, -0.0105435, 0.0108632
6: -0.0069032, -0.0007610, -0.0068169, -0.0009576, -0.0059456, 0.0060559
7: -0.0110932, 0.0006184, -0.0110273, 0.0004206, -0.0115138, 0.0116457
8: -0.0110528, 0.0006045, -0.0112769, 0.0001720, -0.0112249, 0.0118814
9: 0.9809167, 0.9984845, 0.9810164, 0.9977967, -0.0168800, 0.0174681

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111377, upper bound: 0.0124086
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111377, upper bound: 0.0124086
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055008, 0.0055654, -0.0051373, 0.0059815, -0.0110137, 0.0103213
1: 0.0019014, 0.0122909, 0.0019422, 0.0118258, -0.0099243, 0.0103487
2: 0.0140902, 0.0309018, 0.0146924, 0.0316475, -0.0149069, 0.0138728
3: -0.0099040, -0.0021854, -0.0101021, -0.0024892, -0.0074148, 0.0079167
4: -0.0022233, 0.0026880, -0.0020490, 0.0024703, -0.0046936, 0.0047370
5: -0.0038318, 0.0071722, -0.0037235, 0.0067224, -0.0105542, 0.0108957
6: -0.0068858, -0.0007401, -0.0068225, -0.0009472, -0.0059387, 0.0060824
7: -0.0111528, 0.0005308, -0.0110654, 0.0004304, -0.0115832, 0.0115962
8: -0.0110558, 0.0006410, -0.0112907, 0.0001849, -0.0112407, 0.0119317
9: 0.9808924, 0.9984529, 0.9809808, 0.9978092, -0.0169169, 0.0174721

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117141, upper bound: 0.0129544
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117141, upper bound: 0.0129544
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055729, 0.0055935, -0.0055086, 0.0054643, -0.0106021, 0.0106982
1: 0.0018327, 0.0123846, 0.0022620, 0.0123909, -0.0105582, 0.0101226
2: 0.0140116, 0.0309130, 0.0142852, 0.0309471, -0.0142787, 0.0140809
3: -0.0099440, -0.0021430, -0.0097854, -0.0021454, -0.0077987, 0.0076425
4: -0.0022622, 0.0027006, -0.0022423, 0.0023532, -0.0046154, 0.0049430
5: -0.0038389, 0.0072948, -0.0034578, 0.0072499, -0.0110889, 0.0107526
6: -0.0069078, -0.0006861, -0.0066674, -0.0007199, -0.0061878, 0.0059813
7: -0.0112056, 0.0006285, -0.0111346, 0.0001052, -0.0113109, 0.0117630
8: -0.0110580, 0.0007402, -0.0109085, 0.0006345, -0.0116925, 0.0116486
9: 0.9807805, 0.9985095, 0.9809152, 0.9975743, -0.0167938, 0.0175943

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112146, upper bound: 0.0126763
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112146, upper bound: 0.0126763
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056130, 0.0055728, -0.0055222, 0.0054834, -0.0106887, 0.0106902
1: 0.0018883, 0.0124472, 0.0022505, 0.0124151, -0.0105269, 0.0101967
2: 0.0139657, 0.0309155, 0.0142708, 0.0309785, -0.0145042, 0.0141106
3: -0.0099077, -0.0020804, -0.0097951, -0.0021241, -0.0077836, 0.0077147
4: -0.0022691, 0.0026951, -0.0022445, 0.0023594, -0.0046285, 0.0049396
5: -0.0038412, 0.0073168, -0.0034685, 0.0072581, -0.0110993, 0.0107854
6: -0.0068905, -0.0006653, -0.0066730, -0.0007098, -0.0061807, 0.0060077
7: -0.0112642, 0.0005409, -0.0111719, 0.0001152, -0.0113793, 0.0117128
8: -0.0110610, 0.0007766, -0.0109222, 0.0006468, -0.0117078, 0.0116988
9: 0.9807566, 0.9984781, 0.9808817, 0.9975876, -0.0168310, 0.0175964

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117537, upper bound: 0.0130438
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117537, upper bound: 0.0130438
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054593, 0.0055861, -0.0050960, 0.0060071, -0.0110112, 0.0103204
1: 0.0018457, 0.0122283, 0.0018868, 0.0118011, -0.0099554, 0.0103415
2: 0.0141371, 0.0308985, 0.0147222, 0.0316817, -0.0148663, 0.0138598
3: -0.0099402, -0.0022487, -0.0101218, -0.0025238, -0.0074164, 0.0078731
4: -0.0022165, 0.0026935, -0.0020521, 0.0025187, -0.0047351, 0.0047456
5: -0.0038299, 0.0071502, -0.0037964, 0.0066987, -0.0105286, 0.0109466
6: -0.0069032, -0.0007610, -0.0068614, -0.0009460, -0.0059572, 0.0061004
7: -0.0110932, 0.0006184, -0.0110757, 0.0004826, -0.0115757, 0.0116941
8: -0.0110528, 0.0006045, -0.0113195, 0.0001481, -0.0112010, 0.0119240
9: 0.9809167, 0.9984845, 0.9809585, 0.9979365, -0.0170198, 0.0175260

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111694, upper bound: 0.0123607
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111694, upper bound: 0.0123607
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055008, 0.0055654, -0.0051101, 0.0060261, -0.0110685, 0.0103133
1: 0.0019014, 0.0122909, 0.0018756, 0.0118278, -0.0099263, 0.0104152
2: 0.0140902, 0.0309018, 0.0147073, 0.0317128, -0.0149784, 0.0138918
3: -0.0099040, -0.0021854, -0.0101313, -0.0025013, -0.0074026, 0.0079459
4: -0.0022233, 0.0026880, -0.0020544, 0.0025248, -0.0047481, 0.0047424
5: -0.0038318, 0.0071722, -0.0038068, 0.0067076, -0.0105394, 0.0109790
6: -0.0068858, -0.0007401, -0.0068669, -0.0009356, -0.0059502, 0.0061268
7: -0.0111528, 0.0005308, -0.0111134, 0.0004922, -0.0116450, 0.0116443
8: -0.0110558, 0.0006410, -0.0113331, 0.0001610, -0.0112168, 0.0119741
9: 0.9808924, 0.9984529, 0.9809233, 0.9979489, -0.0170565, 0.0175296

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117474, upper bound: 0.0128989
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117474, upper bound: 0.0128989
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055438, 0.0056410, -0.0055394, 0.0054222, -0.0105454, 0.0107667
1: 0.0017710, 0.0123958, 0.0023217, 0.0123942, -0.0106232, 0.0100741
2: 0.0140319, 0.0309838, 0.0142653, 0.0308822, -0.0142287, 0.0141380
3: -0.0099711, -0.0021535, -0.0097585, -0.0021359, -0.0078351, 0.0076050
4: -0.0022784, 0.0027625, -0.0022310, 0.0023002, -0.0045786, 0.0049935
5: -0.0039214, 0.0072905, -0.0033898, 0.0072652, -0.0111866, 0.0106803
6: -0.0069499, -0.0006680, -0.0066267, -0.0007314, -0.0062185, 0.0059586
7: -0.0112557, 0.0006882, -0.0110926, 0.0000465, -0.0113022, 0.0117808
8: -0.0110984, 0.0007117, -0.0108721, 0.0006634, -0.0117618, 0.0115837
9: 0.9807070, 0.9986542, 0.9809726, 0.9974478, -0.0167408, 0.0176815

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0128417
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0128417
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055845, 0.0056174, -0.0055529, 0.0054417, -0.0106100, 0.0107555
1: 0.0018241, 0.0124580, 0.0023100, 0.0124183, -0.0105941, 0.0101480
2: 0.0139859, 0.0309797, 0.0142510, 0.0309143, -0.0143859, 0.0141616
3: -0.0099341, -0.0020937, -0.0097684, -0.0021151, -0.0078191, 0.0076748
4: -0.0022850, 0.0027563, -0.0022332, 0.0023064, -0.0045915, 0.0049895
5: -0.0039230, 0.0073117, -0.0034008, 0.0072732, -0.0111963, 0.0107125
6: -0.0069337, -0.0006485, -0.0066324, -0.0007213, -0.0062124, 0.0059839
7: -0.0113149, 0.0006048, -0.0111300, 0.0000567, -0.0113716, 0.0117348
8: -0.0110989, 0.0007474, -0.0108863, 0.0006757, -0.0117747, 0.0116337
9: 0.9806883, 0.9986237, 0.9809389, 0.9974611, -0.0167728, 0.0176848

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117224, upper bound: 0.0131787
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117224, upper bound: 0.0131787
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054297, 0.0056335, -0.0051233, 0.0059621, -0.0109503, 0.0103856
1: 0.0017844, 0.0122391, 0.0019534, 0.0117993, -0.0100149, 0.0102857
2: 0.0141554, 0.0309691, 0.0147073, 0.0316157, -0.0148146, 0.0139163
3: -0.0099673, -0.0022608, -0.0100924, -0.0025114, -0.0074559, 0.0078315
4: -0.0022321, 0.0027554, -0.0020468, 0.0024642, -0.0046963, 0.0048022
5: -0.0039121, 0.0071459, -0.0037130, 0.0067135, -0.0106257, 0.0108589
6: -0.0069455, -0.0007426, -0.0068169, -0.0009576, -0.0059878, 0.0060744
7: -0.0111453, 0.0006781, -0.0110273, 0.0004206, -0.0115659, 0.0117055
8: -0.0110932, 0.0005761, -0.0112769, 0.0001720, -0.0112652, 0.0118530
9: 0.9808418, 0.9986300, 0.9810164, 0.9977967, -0.0169548, 0.0176136

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0125452
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0125452
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054714, 0.0056101, -0.0051373, 0.0059815, -0.0110089, 0.0103747
1: 0.0018372, 0.0123032, 0.0019422, 0.0118258, -0.0099886, 0.0103610
2: 0.0141084, 0.0309659, 0.0146924, 0.0316475, -0.0149254, 0.0139424
3: -0.0099304, -0.0021984, -0.0101021, -0.0024892, -0.0074412, 0.0079037
4: -0.0022387, 0.0027493, -0.0020490, 0.0024703, -0.0047090, 0.0047983
5: -0.0039135, 0.0071675, -0.0037235, 0.0067224, -0.0106360, 0.0108909
6: -0.0069291, -0.0007225, -0.0068225, -0.0009472, -0.0059820, 0.0061000
7: -0.0112057, 0.0005947, -0.0110654, 0.0004304, -0.0116362, 0.0116601
8: -0.0110937, 0.0006124, -0.0112907, 0.0001849, -0.0112786, 0.0119031
9: 0.9808234, 0.9985991, 0.9809808, 0.9978092, -0.0169858, 0.0176184

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117130, upper bound: 0.0130346
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117130, upper bound: 0.0130346
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055438, 0.0056410, -0.0055086, 0.0054643, -0.0106191, 0.0107764
1: 0.0017710, 0.0123958, 0.0022620, 0.0123909, -0.0106199, 0.0101339
2: 0.0140319, 0.0309838, 0.0142852, 0.0309471, -0.0144320, 0.0142851
3: -0.0099711, -0.0021535, -0.0097854, -0.0021454, -0.0078257, 0.0076319
4: -0.0022784, 0.0027625, -0.0022423, 0.0023532, -0.0046316, 0.0050048
5: -0.0039214, 0.0072905, -0.0034578, 0.0072499, -0.0111714, 0.0107484
6: -0.0069499, -0.0006680, -0.0066674, -0.0007199, -0.0062300, 0.0059993
7: -0.0112557, 0.0006882, -0.0111346, 0.0001052, -0.0113609, 0.0118228
8: -0.0110984, 0.0007117, -0.0109085, 0.0006345, -0.0117329, 0.0116201
9: 0.9807070, 0.9986542, 0.9809152, 0.9975743, -0.0168673, 0.0177390

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111820, upper bound: 0.0128417
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111820, upper bound: 0.0128417
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055845, 0.0056174, -0.0055222, 0.0054834, -0.0106813, 0.0107660
1: 0.0018241, 0.0124580, 0.0022505, 0.0124151, -0.0105910, 0.0102075
2: 0.0139859, 0.0309797, 0.0142708, 0.0309785, -0.0145801, 0.0143143
3: -0.0099341, -0.0020937, -0.0097951, -0.0021241, -0.0078100, 0.0077014
4: -0.0022850, 0.0027563, -0.0022445, 0.0023594, -0.0046444, 0.0050008
5: -0.0039230, 0.0073117, -0.0034685, 0.0072581, -0.0111811, 0.0107803
6: -0.0069337, -0.0006485, -0.0066730, -0.0007098, -0.0062239, 0.0060245
7: -0.0113149, 0.0006048, -0.0111719, 0.0001152, -0.0114301, 0.0117767
8: -0.0110989, 0.0007474, -0.0109222, 0.0006468, -0.0117458, 0.0116696
9: 0.9806883, 0.9986237, 0.9808817, 0.9975876, -0.0168993, 0.0177420

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131787
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131787
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054297, 0.0056335, -0.0050960, 0.0060071, -0.0110317, 0.0103985
1: 0.0017844, 0.0122391, 0.0018868, 0.0118011, -0.0100167, 0.0103523
2: 0.0141554, 0.0309691, 0.0147222, 0.0316817, -0.0150295, 0.0140573
3: -0.0099673, -0.0022608, -0.0101218, -0.0025238, -0.0074435, 0.0078610
4: -0.0022321, 0.0027554, -0.0020521, 0.0025187, -0.0047507, 0.0048075
5: -0.0039121, 0.0071459, -0.0037964, 0.0066987, -0.0106108, 0.0109423
6: -0.0069455, -0.0007426, -0.0068614, -0.0009460, -0.0059994, 0.0061188
7: -0.0111453, 0.0006781, -0.0110757, 0.0004826, -0.0116279, 0.0117538
8: -0.0110932, 0.0005761, -0.0113195, 0.0001481, -0.0112413, 0.0118956
9: 0.9808418, 0.9986300, 0.9809585, 0.9979365, -0.0170947, 0.0176715

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111386, upper bound: 0.0125452
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111386, upper bound: 0.0125452
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054714, 0.0056101, -0.0051101, 0.0060261, -0.0110889, 0.0103883
1: 0.0018372, 0.0123032, 0.0018756, 0.0118278, -0.0099906, 0.0104276
2: 0.0141084, 0.0309659, 0.0147073, 0.0317128, -0.0151305, 0.0140893
3: -0.0099304, -0.0021984, -0.0101313, -0.0025013, -0.0074291, 0.0079329
4: -0.0022387, 0.0027493, -0.0020544, 0.0025248, -0.0047635, 0.0048037
5: -0.0039135, 0.0071675, -0.0038068, 0.0067076, -0.0106211, 0.0109743
6: -0.0069291, -0.0007225, -0.0068669, -0.0009356, -0.0059935, 0.0061444
7: -0.0112057, 0.0005947, -0.0111134, 0.0004922, -0.0116980, 0.0117081
8: -0.0110937, 0.0006124, -0.0113331, 0.0001610, -0.0112547, 0.0119455
9: 0.9808234, 0.9985991, 0.9809233, 0.9979489, -0.0171255, 0.0176758

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117132, upper bound: 0.0130346
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117132, upper bound: 0.0130346
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054324, 0.0055750, -0.0055729, 0.0055935, -0.0106836, 0.0107934
1: 0.0018992, 0.0122615, 0.0018327, 0.0123846, -0.0104854, 0.0104288
2: 0.0141860, 0.0309130, 0.0140116, 0.0309130, -0.0143449, 0.0144577
3: -0.0099098, -0.0022315, -0.0099440, -0.0021430, -0.0077668, 0.0077126
4: -0.0022006, 0.0026872, -0.0022622, 0.0027006, -0.0049012, 0.0049495
5: -0.0038358, 0.0071124, -0.0038389, 0.0072948, -0.0111306, 0.0109514
6: -0.0068882, -0.0007664, -0.0069078, -0.0006861, -0.0062021, 0.0061414
7: -0.0111060, 0.0005338, -0.0112056, 0.0006285, -0.0117345, 0.0117394
8: -0.0110637, 0.0005569, -0.0110580, 0.0007402, -0.0118039, 0.0116150
9: 0.9809830, 0.9984475, 0.9807805, 0.9985095, -0.0175264, 0.0176670

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120697, upper bound: 0.0126777
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120697, upper bound: 0.0126777
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054456, 0.0055944, -0.0056130, 0.0055728, -0.0106744, 0.0108961
1: 0.0018874, 0.0122853, 0.0018883, 0.0124472, -0.0105598, 0.0103970
2: 0.0141717, 0.0309453, 0.0139657, 0.0309155, -0.0143677, 0.0148069
3: -0.0099198, -0.0022108, -0.0099077, -0.0020804, -0.0078394, 0.0076969
4: -0.0022028, 0.0026934, -0.0022691, 0.0026951, -0.0048979, 0.0049625
5: -0.0038469, 0.0071203, -0.0038412, 0.0073168, -0.0111638, 0.0109615
6: -0.0068940, -0.0007561, -0.0068905, -0.0006653, -0.0062287, 0.0061344
7: -0.0111430, 0.0005440, -0.0112642, 0.0005409, -0.0116839, 0.0118082
8: -0.0110778, 0.0005692, -0.0110610, 0.0007766, -0.0118544, 0.0116302
9: 0.9809487, 0.9984610, 0.9807566, 0.9984781, -0.0175293, 0.0177044

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122316, upper bound: 0.0129858
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122316, upper bound: 0.0129858
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0050188, 0.0061196, -0.0054593, 0.0055861, -0.0102939, 0.0112000
1: 0.0015184, 0.0116790, 0.0018457, 0.0122283, -0.0107099, 0.0098332
2: 0.0146289, 0.0316473, 0.0141371, 0.0308985, -0.0141245, 0.0150345
3: -0.0102465, -0.0026087, -0.0099402, -0.0022487, -0.0079978, 0.0073314
4: -0.0020214, 0.0028451, -0.0022165, 0.0026935, -0.0047149, 0.0050616
5: -0.0041634, 0.0065634, -0.0038299, 0.0071502, -0.0113136, 0.0103934
6: -0.0070819, -0.0009847, -0.0069032, -0.0007610, -0.0063208, 0.0059186
7: -0.0110469, 0.0009164, -0.0110932, 0.0006184, -0.0116653, 0.0120096
8: -0.0114731, 0.0000679, -0.0110528, 0.0006045, -0.0120776, 0.0111208
9: 0.9810241, 0.9987798, 0.9809167, 0.9984845, -0.0174603, 0.0178631

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118901, upper bound: 0.0126777
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118901, upper bound: 0.0126777
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0050327, 0.0061389, -0.0055008, 0.0055654, -0.0102857, 0.0112613
1: 0.0015071, 0.0117046, 0.0019014, 0.0122909, -0.0107837, 0.0098031
2: 0.0146140, 0.0316795, 0.0140902, 0.0309018, -0.0141501, 0.0151560
3: -0.0102562, -0.0025867, -0.0099040, -0.0021854, -0.0080708, 0.0073172
4: -0.0020237, 0.0028512, -0.0022233, 0.0026880, -0.0047117, 0.0050745
5: -0.0041739, 0.0065721, -0.0038318, 0.0071722, -0.0113461, 0.0104039
6: -0.0070874, -0.0009741, -0.0068858, -0.0007401, -0.0063473, 0.0059117
7: -0.0110848, 0.0009264, -0.0111528, 0.0005308, -0.0116157, 0.0120792
8: -0.0114869, 0.0000807, -0.0110558, 0.0006410, -0.0121279, 0.0111365
9: 0.9809882, 0.9987926, 0.9808924, 0.9984529, -0.0174647, 0.0179003

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121004, upper bound: 0.0129858
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121004, upper bound: 0.0129858
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0053701, 0.0055777, -0.0056071, 0.0056361, -0.0106799, 0.0108491
1: 0.0018563, 0.0121564, 0.0018104, 0.0125030, -0.0106468, 0.0103460
2: 0.0142526, 0.0308812, 0.0139642, 0.0310090, -0.0143531, 0.0145393
3: -0.0099363, -0.0023272, -0.0099442, -0.0020584, -0.0078779, 0.0076169
4: -0.0021921, 0.0026863, -0.0022865, 0.0027631, -0.0049552, 0.0049728
5: -0.0038225, 0.0070810, -0.0039355, 0.0073219, -0.0111444, 0.0110164
6: -0.0068995, -0.0007996, -0.0069401, -0.0006355, -0.0062640, 0.0061405
7: -0.0109985, 0.0006102, -0.0113643, 0.0006164, -0.0116149, 0.0119745
8: -0.0110474, 0.0005026, -0.0111128, 0.0007662, -0.0118136, 0.0116153
9: 0.9810389, 0.9984642, 0.9806548, 0.9986399, -0.0176010, 0.0178095

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121784, upper bound: 0.0125652
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121784, upper bound: 0.0125652
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0054090, 0.0055572, -0.0056210, 0.0056552, -0.0107440, 0.0108761
1: 0.0019117, 0.0122140, 0.0017986, 0.0125278, -0.0106161, 0.0104155
2: 0.0142091, 0.0308843, 0.0139490, 0.0310403, -0.0145268, 0.0147424
3: -0.0099003, -0.0022676, -0.0099538, -0.0020374, -0.0078629, 0.0076862
4: -0.0021989, 0.0026808, -0.0022889, 0.0027694, -0.0049683, 0.0049697
5: -0.0038244, 0.0071021, -0.0039464, 0.0073304, -0.0111548, 0.0110486
6: -0.0068822, -0.0007798, -0.0069459, -0.0006249, -0.0062573, 0.0061661
7: -0.0110541, 0.0005229, -0.0114023, 0.0006266, -0.0116807, 0.0119252
8: -0.0110505, 0.0005376, -0.0111265, 0.0007791, -0.0118296, 0.0116641
9: 0.9810173, 0.9984320, 0.9806208, 0.9986536, -0.0176363, 0.0178112

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122624, upper bound: 0.0129305
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122624, upper bound: 0.0129305
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0050188, 0.0061196, -0.0054297, 0.0056335, -0.0103509, 0.0111967
1: 0.0015184, 0.0116790, 0.0017844, 0.0122391, -0.0107207, 0.0098946
2: 0.0146289, 0.0316473, 0.0141554, 0.0309691, -0.0141997, 0.0150566
3: -0.0102465, -0.0026087, -0.0099673, -0.0022608, -0.0079856, 0.0073585
4: -0.0020214, 0.0028451, -0.0022321, 0.0027554, -0.0047768, 0.0050772
5: -0.0041634, 0.0065634, -0.0039121, 0.0071459, -0.0113093, 0.0104756
6: -0.0070819, -0.0009847, -0.0069455, -0.0007426, -0.0063393, 0.0059608
7: -0.0110469, 0.0009164, -0.0111453, 0.0006781, -0.0117250, 0.0120618
8: -0.0114731, 0.0000679, -0.0110932, 0.0005761, -0.0120492, 0.0111611
9: 0.9810241, 0.9987798, 0.9808418, 0.9986300, -0.0176058, 0.0179380

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119464, upper bound: 0.0126583
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119464, upper bound: 0.0126583
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0050327, 0.0061389, -0.0054714, 0.0056101, -0.0103394, 0.0112582
1: 0.0015071, 0.0117046, 0.0018372, 0.0123032, -0.0107961, 0.0098674
2: 0.0146140, 0.0316795, 0.0141084, 0.0309659, -0.0142195, 0.0151749
3: -0.0102562, -0.0025867, -0.0099304, -0.0021984, -0.0080578, 0.0073437
4: -0.0020237, 0.0028512, -0.0022387, 0.0027493, -0.0047730, 0.0050899
5: -0.0041739, 0.0065721, -0.0039135, 0.0071675, -0.0113414, 0.0104856
6: -0.0070874, -0.0009741, -0.0069291, -0.0007225, -0.0063649, 0.0059550
7: -0.0110848, 0.0009264, -0.0112057, 0.0005947, -0.0116795, 0.0121321
8: -0.0114869, 0.0000807, -0.0110937, 0.0006124, -0.0120993, 0.0111744
9: 0.9809882, 0.9987926, 0.9808234, 0.9985991, -0.0176109, 0.0179693

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121436, upper bound: 0.0129305
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121436, upper bound: 0.0129305
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0056071, 0.0056361, -0.0053701, 0.0055777, -0.0108491, 0.0106799
1: 0.0018104, 0.0125030, 0.0018563, 0.0121564, -0.0103460, 0.0106468
2: 0.0139642, 0.0310090, 0.0142526, 0.0308812, -0.0145393, 0.0143531
3: -0.0099442, -0.0020584, -0.0099363, -0.0023272, -0.0076169, 0.0078779
4: -0.0022865, 0.0027631, -0.0021921, 0.0026863, -0.0049728, 0.0049552
5: -0.0039355, 0.0073219, -0.0038225, 0.0070810, -0.0110164, 0.0111444
6: -0.0069401, -0.0006355, -0.0068995, -0.0007996, -0.0061405, 0.0062640
7: -0.0113643, 0.0006164, -0.0109985, 0.0006102, -0.0119745, 0.0116149
8: -0.0111128, 0.0007662, -0.0110474, 0.0005026, -0.0116153, 0.0118136
9: 0.9806548, 0.9986399, 0.9810389, 0.9984642, -0.0178095, 0.0176010

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118951, upper bound: 0.0130875
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118951, upper bound: 0.0130875
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0056210, 0.0056552, -0.0054090, 0.0055572, -0.0108761, 0.0107440
1: 0.0017986, 0.0125278, 0.0019117, 0.0122140, -0.0104155, 0.0106161
2: 0.0139490, 0.0310403, 0.0142091, 0.0308843, -0.0147424, 0.0145268
3: -0.0099538, -0.0020374, -0.0099003, -0.0022676, -0.0076862, 0.0078629
4: -0.0022889, 0.0027694, -0.0021989, 0.0026808, -0.0049697, 0.0049683
5: -0.0039464, 0.0073304, -0.0038244, 0.0071021, -0.0110486, 0.0111548
6: -0.0069459, -0.0006249, -0.0068822, -0.0007798, -0.0061661, 0.0062573
7: -0.0114023, 0.0006266, -0.0110541, 0.0005229, -0.0119252, 0.0116807
8: -0.0111265, 0.0007791, -0.0110505, 0.0005376, -0.0116641, 0.0118296
9: 0.9806208, 0.9986536, 0.9810173, 0.9984320, -0.0178112, 0.0176363

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0131869
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0131869
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054297, 0.0056335, -0.0050188, 0.0061196, -0.0111967, 0.0103509
1: 0.0017844, 0.0122391, 0.0015184, 0.0116790, -0.0098946, 0.0107207
2: 0.0141554, 0.0309691, 0.0146289, 0.0316473, -0.0150566, 0.0141997
3: -0.0099673, -0.0022608, -0.0102465, -0.0026087, -0.0073585, 0.0079856
4: -0.0022321, 0.0027554, -0.0020214, 0.0028451, -0.0050772, 0.0047768
5: -0.0039121, 0.0071459, -0.0041634, 0.0065634, -0.0104756, 0.0113093
6: -0.0069455, -0.0007426, -0.0070819, -0.0009847, -0.0059608, 0.0063393
7: -0.0111453, 0.0006781, -0.0110469, 0.0009164, -0.0120618, 0.0117250
8: -0.0110932, 0.0005761, -0.0114731, 0.0000679, -0.0111611, 0.0120492
9: 0.9808418, 0.9986300, 0.9810241, 0.9987798, -0.0179380, 0.0176058

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118609, upper bound: 0.0127118
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118609, upper bound: 0.0127118
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054714, 0.0056101, -0.0050327, 0.0061389, -0.0112582, 0.0103394
1: 0.0018372, 0.0123032, 0.0015071, 0.0117046, -0.0098674, 0.0107961
2: 0.0141084, 0.0309659, 0.0146140, 0.0316795, -0.0151749, 0.0142195
3: -0.0099304, -0.0021984, -0.0102562, -0.0025867, -0.0073437, 0.0080578
4: -0.0022387, 0.0027493, -0.0020237, 0.0028512, -0.0050899, 0.0047730
5: -0.0039135, 0.0071675, -0.0041739, 0.0065721, -0.0104856, 0.0113414
6: -0.0069291, -0.0007225, -0.0070874, -0.0009741, -0.0059550, 0.0063649
7: -0.0112057, 0.0005947, -0.0110848, 0.0009264, -0.0121321, 0.0116795
8: -0.0110937, 0.0006124, -0.0114869, 0.0000807, -0.0111744, 0.0120993
9: 0.9808234, 0.9985991, 0.9809882, 0.9987926, -0.0179693, 0.0176109

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054195, 0.0056394, -0.0054195, 0.0056394, -0.0107638, 0.0107638
1: 0.0018224, 0.0122972, 0.0018224, 0.0122972, -0.0104748, 0.0104748
2: 0.0141922, 0.0310094, 0.0141922, 0.0310094, -0.0146516, 0.0146516
3: -0.0099463, -0.0022222, -0.0099463, -0.0022222, -0.0077241, 0.0077241
4: -0.0022188, 0.0027551, -0.0022188, 0.0027551, -0.0049739, 0.0049739
5: -0.0039291, 0.0071198, -0.0039291, 0.0071198, -0.0110489, 0.0110489
6: -0.0069377, -0.0007384, -0.0069377, -0.0007384, -0.0061993, 0.0061993
7: -0.0111910, 0.0006083, -0.0111910, 0.0006083, -0.0117993, 0.0117993
8: -0.0111158, 0.0005438, -0.0111158, 0.0005438, -0.0116596, 0.0116596
9: 0.9808795, 0.9986090, 0.9808795, 0.9986090, -0.0177295, 0.0177295

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121555, upper bound: 0.0127314
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122307, upper bound: 0.0130680
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054195, 0.0056394, -0.0050081, 0.0061870, -0.0112981, 0.0103280
1: 0.0018224, 0.0122972, 0.0014401, 0.0117137, -0.0098913, 0.0108571
2: 0.0141922, 0.0310094, 0.0146289, 0.0317444, -0.0153661, 0.0141655
3: -0.0099463, -0.0022222, -0.0102853, -0.0025958, -0.0073505, 0.0080631
4: -0.0022188, 0.0027551, -0.0020369, 0.0029138, -0.0051326, 0.0047919
5: -0.0039291, 0.0071198, -0.0042614, 0.0065708, -0.0104999, 0.0113812
6: -0.0069377, -0.0007384, -0.0071326, -0.0009575, -0.0059801, 0.0063942
7: -0.0111910, 0.0006083, -0.0111388, 0.0009869, -0.0121779, 0.0117472
8: -0.0111158, 0.0005438, -0.0115292, 0.0000599, -0.0111757, 0.0120730
9: 0.9808795, 0.9986090, 0.9809195, 0.9989452, -0.0180658, 0.0176895

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121555, upper bound: 0.0127314
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122307, upper bound: 0.0130680
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0049943, 0.0061682, -0.0054297, 0.0056335, -0.0103693, 0.0112812
1: 0.0014511, 0.0116880, 0.0017844, 0.0122391, -0.0107880, 0.0099036
2: 0.0146437, 0.0317131, 0.0141554, 0.0309691, -0.0143567, 0.0152775
3: -0.0102759, -0.0026180, -0.0099673, -0.0022608, -0.0080150, 0.0073493
4: -0.0020345, 0.0029078, -0.0022321, 0.0027554, -0.0047899, 0.0051398
5: -0.0042511, 0.0065622, -0.0039121, 0.0071459, -0.0113970, 0.0104743
6: -0.0071271, -0.0009679, -0.0069455, -0.0007426, -0.0063846, 0.0059775
7: -0.0111015, 0.0009772, -0.0111453, 0.0006781, -0.0117796, 0.0121226
8: -0.0115158, 0.0000472, -0.0110932, 0.0005761, -0.0120919, 0.0111404
9: 0.9809551, 0.9989324, 0.9808418, 0.9986300, -0.0176749, 0.0180906

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118904, upper bound: 0.0127790
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118904, upper bound: 0.0127790
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0050081, 0.0061870, -0.0054714, 0.0056101, -0.0103589, 0.0113406
1: 0.0014401, 0.0117137, 0.0018372, 0.0123032, -0.0108631, 0.0098765
2: 0.0146289, 0.0317444, 0.0141084, 0.0309659, -0.0143821, 0.0153841
3: -0.0102853, -0.0025958, -0.0099304, -0.0021984, -0.0080869, 0.0073346
4: -0.0020369, 0.0029138, -0.0022387, 0.0027493, -0.0047861, 0.0051525
5: -0.0042614, 0.0065708, -0.0039135, 0.0071675, -0.0114289, 0.0104843
6: -0.0071326, -0.0009575, -0.0069291, -0.0007225, -0.0064100, 0.0059716
7: -0.0111388, 0.0009869, -0.0112057, 0.0005947, -0.0117335, 0.0121927
8: -0.0115292, 0.0000599, -0.0110937, 0.0006124, -0.0121416, 0.0111536
9: 0.9809195, 0.9989452, 0.9808234, 0.9985991, -0.0176796, 0.0181218

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
time: 1.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
time: 1.15 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.52 seconds
NS_A1_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118339, upper bound: 0.0114395
NS_A1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120331, upper bound: 0.0117351
NS_A1_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118339, upper bound: 0.0114395
NS_A1_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120331, upper bound: 0.0117351
NS_A1_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117230, upper bound: 0.0117230
NS_A1_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117230, upper bound: 0.0117230
NS_A1_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0114340, upper bound: 0.0118656
NS_A1_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117339, upper bound: 0.0120764
NS_A1_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0114340, upper bound: 0.0118656
NS_A1_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117339, upper bound: 0.0120764
NS_A1_B1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117210, upper bound: 0.0117562
NS_A1_B1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117210, upper bound: 0.0117562
NS_A1_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0127214, upper bound: 0.0111808
NS_A1_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0131007, upper bound: 0.0117227
NS_A1_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0129654, upper bound: 0.0114347
NS_A1_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0131007, upper bound: 0.0117227
NS_A1_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0111377
NS_A1_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0124086, upper bound: 0.0111377
NS_A1_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117141
NS_A1_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0129544, upper bound: 0.0117141
NS_A1_B1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0126763, upper bound: 0.0112146
NS_A1_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
NS_A1_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0129288, upper bound: 0.0114822
NS_A1_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
NS_A1_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0123607, upper bound: 0.0111694
NS_A1_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0123607, upper bound: 0.0111694
NS_A1_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
NS_A1_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
NS_A1_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118656, upper bound: 0.0114340
NS_A1_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120764, upper bound: 0.0117339
NS_A1_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118656, upper bound: 0.0114340
NS_A1_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120764, upper bound: 0.0117339
NS_A1_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117562, upper bound: 0.0117210
NS_A1_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117562, upper bound: 0.0117210
NS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118371, upper bound: 0.0114887
NS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120315, upper bound: 0.0117643
NS_A1_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118371, upper bound: 0.0114887
NS_A1_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120315, upper bound: 0.0117643
NS_A1_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117215, upper bound: 0.0117562
NS_A1_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117215, upper bound: 0.0117562
NS_A1_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0128417, upper bound: 0.0111805
NS_A1_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0131787, upper bound: 0.0117224
NS_A1_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130589, upper bound: 0.0114297
NS_A1_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0131787, upper bound: 0.0117224
NS_A1_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0125452, upper bound: 0.0111372
NS_A1_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0125452, upper bound: 0.0111372
NS_A1_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117130
NS_A1_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130346, upper bound: 0.0117130
NS_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0126830, upper bound: 0.0112146
NS_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
NS_A1_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0129302, upper bound: 0.0114822
NS_A1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0130438, upper bound: 0.0117537
NS_A1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0123631, upper bound: 0.0111694
NS_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0123631, upper bound: 0.0111694
NS_A1_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
NS_A1_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0128989, upper bound: 0.0117474
NS_A2_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111808, upper bound: 0.0127214
NS_A2_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111808, upper bound: 0.0127214
NS_A2_B1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131007
NS_A2_B1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131007
NS_A2_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111377, upper bound: 0.0124086
NS_A2_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111377, upper bound: 0.0124086
NS_A2_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117141, upper bound: 0.0129544
NS_A2_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117141, upper bound: 0.0129544
NS_A2_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0112146, upper bound: 0.0126763
NS_A2_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0112146, upper bound: 0.0126763
NS_A2_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117537, upper bound: 0.0130438
NS_A2_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117537, upper bound: 0.0130438
NS_A2_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111694, upper bound: 0.0123607
NS_A2_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111694, upper bound: 0.0123607
NS_A2_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117474, upper bound: 0.0128989
NS_A2_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117474, upper bound: 0.0128989
NS_A2_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0128417
NS_A2_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0128417
NS_A2_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117224, upper bound: 0.0131787
NS_A2_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117224, upper bound: 0.0131787
NS_A2_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0125452
NS_A2_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111372, upper bound: 0.0125452
NS_A2_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117130, upper bound: 0.0130346
NS_A2_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117130, upper bound: 0.0130346
NS_A2_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111820, upper bound: 0.0128417
NS_A2_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111820, upper bound: 0.0128417
NS_A2_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131787
NS_A2_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117227, upper bound: 0.0131787
NS_A2_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111386, upper bound: 0.0125452
NS_A2_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0111386, upper bound: 0.0125452
NS_A2_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117132, upper bound: 0.0130346
NS_A2_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0117132, upper bound: 0.0130346
NS_A2_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120697, upper bound: 0.0126777
NS_A2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120697, upper bound: 0.0126777
NS_A2_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0122316, upper bound: 0.0129858
NS_A2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0122316, upper bound: 0.0129858
NS_A2_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118901, upper bound: 0.0126777
NS_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118901, upper bound: 0.0126777
NS_A2_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121004, upper bound: 0.0129858
NS_A2_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121004, upper bound: 0.0129858
NS_A2_B2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121784, upper bound: 0.0125652
NS_A2_B2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121784, upper bound: 0.0125652
NS_A2_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0122624, upper bound: 0.0129305
NS_A2_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0122624, upper bound: 0.0129305
NS_A2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0119464, upper bound: 0.0126583
NS_A2_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0119464, upper bound: 0.0126583
NS_A2_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121436, upper bound: 0.0129305
NS_A2_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121436, upper bound: 0.0129305
NS_A2_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118951, upper bound: 0.0130875
NS_A2_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118951, upper bound: 0.0130875
NS_A2_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0131869
NS_A2_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0131869
NS_A2_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118609, upper bound: 0.0127118
NS_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118609, upper bound: 0.0127118
NS_A2_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
NS_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121555, upper bound: 0.0127314
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0122307, upper bound: 0.0130680
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0121555, upper bound: 0.0127314
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0122307, upper bound: 0.0130680
NS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118904, upper bound: 0.0127790
NS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0118904, upper bound: 0.0127790
NS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680
NS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 9, lower bound: -0.0120964, upper bound: 0.0130680

## BFS NS instance: NS_A1_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054756, 0.0054251, -0.0055394, 0.0054222, -0.0104332, 0.0104928
1: 0.0022835, 0.0122868, 0.0023217, 0.0123942, -0.0101107, 0.0099650
2: 0.0143318, 0.0308501, 0.0142653, 0.0308822, -0.0139206, 0.0139723
3: -0.0097814, -0.0022327, -0.0097585, -0.0021359, -0.0076455, 0.0075258
4: -0.0022229, 0.0022993, -0.0022310, 0.0023002, -0.0045231, 0.0045303
5: -0.0033758, 0.0072324, -0.0033898, 0.0072652, -0.0106410, 0.0106222
6: -0.0066369, -0.0007657, -0.0066267, -0.0007314, -0.0059055, 0.0058610
7: -0.0109823, 0.0001146, -0.0110926, 0.0000465, -0.0110288, 0.0112072
8: -0.0108561, 0.0006082, -0.0108721, 0.0006634, -0.0115195, 0.0114803
9: 0.9810299, 0.9974655, 0.9809726, 0.9974478, -0.0164179, 0.0164928

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121687, upper bound: 0.0121687
time: 1.14 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121687, upper bound: 0.0121743
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055163, 0.0054050, -0.0055529, 0.0054417, -0.0104980, 0.0104836
1: 0.0023342, 0.0123476, 0.0023100, 0.0124183, -0.0100840, 0.0100376
2: 0.0142882, 0.0308556, 0.0142510, 0.0309143, -0.0140882, 0.0140068
3: -0.0097493, -0.0021712, -0.0097684, -0.0021151, -0.0076342, 0.0075972
4: -0.0022296, 0.0022939, -0.0022332, 0.0023064, -0.0045361, 0.0045271
5: -0.0033783, 0.0072546, -0.0034008, 0.0072732, -0.0106516, 0.0106554
6: -0.0066207, -0.0007445, -0.0066324, -0.0007213, -0.0058995, 0.0058879
7: -0.0110440, 0.0000358, -0.0111300, 0.0000567, -0.0111007, 0.0111658
8: -0.0108593, 0.0006442, -0.0108863, 0.0006757, -0.0115350, 0.0115305
9: 0.9810047, 0.9974326, 0.9809389, 0.9974611, -0.0164564, 0.0164937

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121743, upper bound: 0.0122126
time: 1.06 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121743, upper bound: 0.0122774
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054756, 0.0054251, -0.0051233, 0.0059621, -0.0109595, 0.0100640
1: 0.0022835, 0.0122868, 0.0019534, 0.0117993, -0.0095158, 0.0103334
2: 0.0143318, 0.0308501, 0.0147073, 0.0316157, -0.0146462, 0.0134944
3: -0.0097814, -0.0022327, -0.0100924, -0.0025114, -0.0072700, 0.0078597
4: -0.0022229, 0.0022993, -0.0020468, 0.0024642, -0.0046871, 0.0043461
5: -0.0033758, 0.0072324, -0.0037130, 0.0067135, -0.0100894, 0.0109454
6: -0.0066369, -0.0007657, -0.0068169, -0.0009576, -0.0056793, 0.0060513
7: -0.0109823, 0.0001146, -0.0110273, 0.0004206, -0.0114029, 0.0111419
8: -0.0108561, 0.0006082, -0.0112769, 0.0001720, -0.0110281, 0.0118852
9: 0.9810299, 0.9974655, 0.9810164, 0.9977967, -0.0167667, 0.0164491

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114240, upper bound: 0.0107781
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109326, upper bound: 0.0106337
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055163, 0.0054050, -0.0051373, 0.0059815, -0.0110181, 0.0100563
1: 0.0023342, 0.0123476, 0.0019422, 0.0118258, -0.0094915, 0.0104054
2: 0.0142882, 0.0308556, 0.0146924, 0.0316475, -0.0147718, 0.0135342
3: -0.0097493, -0.0021712, -0.0101021, -0.0024892, -0.0072601, 0.0079309
4: -0.0022296, 0.0022939, -0.0020490, 0.0024703, -0.0046999, 0.0043430
5: -0.0033783, 0.0072546, -0.0037235, 0.0067224, -0.0101008, 0.0109780
6: -0.0066207, -0.0007445, -0.0068225, -0.0009472, -0.0056735, 0.0060780
7: -0.0110440, 0.0000358, -0.0110654, 0.0004304, -0.0114744, 0.0111012
8: -0.0108593, 0.0006442, -0.0112907, 0.0001849, -0.0110442, 0.0119350
9: 0.9810047, 0.9974326, 0.9809808, 0.9978092, -0.0168046, 0.0164518

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115878, upper bound: 0.0111808
time: 0.99 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115878, upper bound: 0.0117351
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0055163, 0.0054050, -0.0100563, 0.0110181
1: 0.0019422, 0.0118258, 0.0023342, 0.0123476, -0.0104054, 0.0094915
2: 0.0146924, 0.0316475, 0.0142882, 0.0308556, -0.0135342, 0.0147718
3: -0.0101021, -0.0024892, -0.0097493, -0.0021712, -0.0079309, 0.0072601
4: -0.0020490, 0.0024703, -0.0022296, 0.0022939, -0.0043430, 0.0046999
5: -0.0037235, 0.0067224, -0.0033783, 0.0072546, -0.0109780, 0.0101008
6: -0.0068225, -0.0009472, -0.0066207, -0.0007445, -0.0060780, 0.0056735
7: -0.0110654, 0.0004304, -0.0110440, 0.0000358, -0.0111012, 0.0114744
8: -0.0112907, 0.0001849, -0.0108593, 0.0006442, -0.0119350, 0.0110442
9: 0.9809808, 0.9978092, 0.9810047, 0.9974326, -0.0164518, 0.0168046

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109744, upper bound: 0.0112211
time: 1.41 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0109744, upper bound: 0.0117227
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0051025, 0.0059440, -0.0105573, 0.0105638
1: 0.0019422, 0.0118258, 0.0019660, 0.0117622, -0.0098200, 0.0098597
2: 0.0146924, 0.0316475, 0.0147275, 0.0315883, -0.0141436, 0.0141993
3: -0.0101021, -0.0024892, -0.0100827, -0.0025421, -0.0075600, 0.0075935
4: -0.0020490, 0.0024703, -0.0020455, 0.0024579, -0.0045070, 0.0045158
5: -0.0037235, 0.0067224, -0.0037011, 0.0067033, -0.0104268, 0.0104236
6: -0.0068225, -0.0009472, -0.0068107, -0.0009684, -0.0058541, 0.0058635
7: -0.0110654, 0.0004304, -0.0109877, 0.0004095, -0.0114749, 0.0114182
8: -0.0112907, 0.0001849, -0.0112636, 0.0001554, -0.0114461, 0.0114485
9: 0.9809808, 0.9978092, 0.9810464, 0.9977819, -0.0168011, 0.0167629

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109744, upper bound: 0.0112211
time: 0.89 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0109744, upper bound: 0.0117227
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055086, 0.0054643, -0.0054756, 0.0054251, -0.0104807, 0.0104842
1: 0.0022620, 0.0123909, 0.0022835, 0.0122868, -0.0100248, 0.0101074
2: 0.0142852, 0.0309471, 0.0143318, 0.0308501, -0.0139906, 0.0139911
3: -0.0097854, -0.0021454, -0.0097814, -0.0022327, -0.0075527, 0.0076361
4: -0.0022423, 0.0023532, -0.0022229, 0.0022993, -0.0045416, 0.0045761
5: -0.0034578, 0.0072499, -0.0033758, 0.0072324, -0.0106903, 0.0106258
6: -0.0066674, -0.0007199, -0.0066369, -0.0007657, -0.0059017, 0.0059170
7: -0.0111346, 0.0001052, -0.0109823, 0.0001146, -0.0112491, 0.0110875
8: -0.0109085, 0.0006345, -0.0108561, 0.0006082, -0.0115167, 0.0114906
9: 0.9809152, 0.9975743, 0.9810299, 0.9974655, -0.0165503, 0.0165444

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114900, upper bound: 0.0119038
time: 1.00 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114414, upper bound: 0.0114825
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0055163, 0.0054050, -0.0104719, 0.0105470
1: 0.0022505, 0.0124151, 0.0023342, 0.0123476, -0.0100971, 0.0100809
2: 0.0142708, 0.0309785, 0.0142882, 0.0308556, -0.0140253, 0.0141576
3: -0.0097951, -0.0021241, -0.0097493, -0.0021712, -0.0076239, 0.0076251
4: -0.0022445, 0.0023594, -0.0022296, 0.0022939, -0.0045385, 0.0045890
5: -0.0034685, 0.0072581, -0.0033783, 0.0072546, -0.0107231, 0.0106365
6: -0.0066730, -0.0007098, -0.0066207, -0.0007445, -0.0059285, 0.0059109
7: -0.0111719, 0.0001152, -0.0110440, 0.0000358, -0.0112077, 0.0111591
8: -0.0109222, 0.0006468, -0.0108593, 0.0006442, -0.0115664, 0.0115061
9: 0.9808817, 0.9975876, 0.9810047, 0.9974326, -0.0165509, 0.0165830

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122120, upper bound: 0.0122226
time: 1.10 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122120, upper bound: 0.0123212
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0050960, 0.0060071, -0.0054756, 0.0054251, -0.0100541, 0.0110155
1: 0.0018868, 0.0118011, 0.0022835, 0.0122868, -0.0104000, 0.0095177
2: 0.0147222, 0.0316817, 0.0143318, 0.0308501, -0.0135127, 0.0147187
3: -0.0101218, -0.0025238, -0.0097814, -0.0022327, -0.0078891, 0.0072576
4: -0.0020521, 0.0025187, -0.0022229, 0.0022993, -0.0043514, 0.0047416
5: -0.0037964, 0.0066987, -0.0033758, 0.0072324, -0.0110289, 0.0100745
6: -0.0068614, -0.0009460, -0.0066369, -0.0007657, -0.0060958, 0.0056909
7: -0.0110757, 0.0004826, -0.0109823, 0.0001146, -0.0111902, 0.0114648
8: -0.0113195, 0.0001481, -0.0108561, 0.0006082, -0.0119278, 0.0110042
9: 0.9809585, 0.9979365, 0.9810299, 0.9974655, -0.0165070, 0.0169066

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107781, upper bound: 0.0114817
time: 0.94 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106335, upper bound: 0.0109613
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0055163, 0.0054050, -0.0100466, 0.0110729
1: 0.0018756, 0.0118278, 0.0023342, 0.0123476, -0.0104720, 0.0094936
2: 0.0147073, 0.0317128, 0.0142882, 0.0308556, -0.0135523, 0.0148433
3: -0.0101313, -0.0025013, -0.0097493, -0.0021712, -0.0079601, 0.0072479
4: -0.0020544, 0.0025248, -0.0022296, 0.0022939, -0.0043483, 0.0047544
5: -0.0038068, 0.0067076, -0.0033783, 0.0072546, -0.0110614, 0.0100859
6: -0.0068669, -0.0009356, -0.0066207, -0.0007445, -0.0061224, 0.0056851
7: -0.0111134, 0.0004922, -0.0110440, 0.0000358, -0.0111492, 0.0115362
8: -0.0113331, 0.0001610, -0.0108593, 0.0006442, -0.0119773, 0.0110203
9: 0.9809233, 0.9979489, 0.9810047, 0.9974326, -0.0165093, 0.0169442

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0116337
time: 1.11 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111805, upper bound: 0.0120764
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0054858, 0.0054465, -0.0051373, 0.0059815, -0.0110048, 0.0101069
1: 0.0022751, 0.0123450, 0.0019422, 0.0118258, -0.0095507, 0.0104028
2: 0.0143075, 0.0309194, 0.0146924, 0.0316475, -0.0147892, 0.0136034
3: -0.0097758, -0.0021798, -0.0101021, -0.0024892, -0.0072866, 0.0079223
4: -0.0022410, 0.0023465, -0.0020490, 0.0024703, -0.0047113, 0.0043955
5: -0.0034458, 0.0072394, -0.0037235, 0.0067224, -0.0101682, 0.0109628
6: -0.0066610, -0.0007330, -0.0068225, -0.0009472, -0.0057138, 0.0060895
7: -0.0110861, 0.0000939, -0.0110654, 0.0004304, -0.0115165, 0.0111593
8: -0.0108951, 0.0006155, -0.0112907, 0.0001849, -0.0110799, 0.0119062
9: 0.9809476, 0.9975584, 0.9809808, 0.9978092, -0.0168617, 0.0165776

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112186, upper bound: 0.0111694
time: 1.14 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112186, upper bound: 0.0111694
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0050755, 0.0059886, -0.0051373, 0.0059815, -0.0105551, 0.0106115
1: 0.0019000, 0.0117650, 0.0019422, 0.0118258, -0.0099257, 0.0098229
2: 0.0147426, 0.0316538, 0.0146924, 0.0316475, -0.0142186, 0.0142141
3: -0.0101119, -0.0025534, -0.0101021, -0.0024892, -0.0076227, 0.0075487
4: -0.0020509, 0.0025120, -0.0020490, 0.0024703, -0.0045212, 0.0045611
5: -0.0037841, 0.0066885, -0.0037235, 0.0067224, -0.0105065, 0.0104119
6: -0.0068549, -0.0009567, -0.0068225, -0.0009472, -0.0059077, 0.0058658
7: -0.0110354, 0.0004710, -0.0110654, 0.0004304, -0.0114659, 0.0115364
8: -0.0113057, 0.0001315, -0.0112907, 0.0001849, -0.0114906, 0.0114223
9: 0.9809884, 0.9979210, 0.9809808, 0.9978092, -0.0168208, 0.0169403

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112186, upper bound: 0.0111694
time: 1.04 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112186, upper bound: 0.0117554
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0055394, 0.0054222, -0.0053701, 0.0055777, -0.0106897, 0.0103614
1: 0.0023217, 0.0123942, 0.0018563, 0.0121564, -0.0098347, 0.0105379
2: 0.0142653, 0.0308822, 0.0142526, 0.0308812, -0.0140138, 0.0139832
3: -0.0097585, -0.0021359, -0.0099363, -0.0023272, -0.0074312, 0.0078004
4: -0.0022310, 0.0023002, -0.0021921, 0.0026863, -0.0049173, 0.0044923
5: -0.0033898, 0.0072652, -0.0038225, 0.0070810, -0.0104708, 0.0110876
6: -0.0066267, -0.0007314, -0.0068995, -0.0007996, -0.0058270, 0.0061680
7: -0.0110926, 0.0000465, -0.0109985, 0.0006102, -0.0117029, 0.0110450
8: -0.0108721, 0.0006634, -0.0110474, 0.0005026, -0.0113746, 0.0117108
9: 0.9809726, 0.9974478, 0.9810389, 0.9984642, -0.0174916, 0.0164089

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130613, upper bound: 0.0120900
time: 1.33 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130613, upper bound: 0.0121228
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055529, 0.0054417, -0.0054090, 0.0055572, -0.0106805, 0.0104255
1: 0.0023100, 0.0124183, 0.0019117, 0.0122140, -0.0099040, 0.0105065
2: 0.0142510, 0.0309143, 0.0142091, 0.0308843, -0.0140444, 0.0141497
3: -0.0097684, -0.0021151, -0.0099003, -0.0022676, -0.0075008, 0.0077852
4: -0.0022332, 0.0023064, -0.0021989, 0.0026808, -0.0049140, 0.0045054
5: -0.0034008, 0.0072732, -0.0038244, 0.0071021, -0.0105029, 0.0110976
6: -0.0066324, -0.0007213, -0.0068822, -0.0007798, -0.0058526, 0.0061610
7: -0.0111300, 0.0000567, -0.0110541, 0.0005229, -0.0116530, 0.0111108
8: -0.0108863, 0.0006757, -0.0110505, 0.0005376, -0.0114239, 0.0117262
9: 0.9809389, 0.9974611, 0.9810173, 0.9984320, -0.0174931, 0.0164438

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131345, upper bound: 0.0120935
time: 1.27 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131345, upper bound: 0.0121755
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054756, 0.0054251, -0.0050188, 0.0061196, -0.0111601, 0.0099843
1: 0.0022835, 0.0122868, 0.0015184, 0.0116790, -0.0093955, 0.0107683
2: 0.0143318, 0.0308501, 0.0146289, 0.0316473, -0.0146919, 0.0135553
3: -0.0097814, -0.0022327, -0.0102465, -0.0026087, -0.0071727, 0.0080138
4: -0.0022229, 0.0022993, -0.0020214, 0.0028451, -0.0050680, 0.0043207
5: -0.0033758, 0.0072324, -0.0041634, 0.0065634, -0.0099393, 0.0113959
6: -0.0066369, -0.0007657, -0.0070819, -0.0009847, -0.0056523, 0.0063162
7: -0.0109823, 0.0001146, -0.0110469, 0.0009164, -0.0118987, 0.0111615
8: -0.0108561, 0.0006082, -0.0114731, 0.0000679, -0.0109241, 0.0120814
9: 0.9810299, 0.9974655, 0.9810241, 0.9987798, -0.0177498, 0.0164413

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127470, upper bound: 0.0107781
time: 1.25 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123792, upper bound: 0.0106364
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055163, 0.0054050, -0.0050327, 0.0061389, -0.0112194, 0.0099765
1: 0.0023342, 0.0123476, 0.0015071, 0.0117046, -0.0093703, 0.0108404
2: 0.0142882, 0.0308556, 0.0146140, 0.0316795, -0.0148184, 0.0135948
3: -0.0097493, -0.0021712, -0.0102562, -0.0025867, -0.0071626, 0.0080850
4: -0.0022296, 0.0022939, -0.0020237, 0.0028512, -0.0050808, 0.0043176
5: -0.0033783, 0.0072546, -0.0041739, 0.0065721, -0.0099504, 0.0114285
6: -0.0066207, -0.0007445, -0.0070874, -0.0009741, -0.0056466, 0.0063429
7: -0.0110440, 0.0000358, -0.0110848, 0.0009264, -0.0119704, 0.0111206
8: -0.0108593, 0.0006442, -0.0114869, 0.0000807, -0.0109400, 0.0121312
9: 0.9810047, 0.9974326, 0.9809882, 0.9987926, -0.0177880, 0.0164444

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127214, upper bound: 0.0111808
time: 1.18 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127214, upper bound: 0.0117227
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0051233, 0.0059621, -0.0053701, 0.0055777, -0.0102610, 0.0108877
1: 0.0019534, 0.0117993, 0.0018563, 0.0121564, -0.0102031, 0.0099430
2: 0.0147073, 0.0316157, 0.0142526, 0.0308812, -0.0135359, 0.0147088
3: -0.0100924, -0.0025114, -0.0099363, -0.0023272, -0.0077651, 0.0074249
4: -0.0020468, 0.0024642, -0.0021921, 0.0026863, -0.0047330, 0.0046563
5: -0.0037130, 0.0067135, -0.0038225, 0.0070810, -0.0107940, 0.0105360
6: -0.0068169, -0.0009576, -0.0068995, -0.0007996, -0.0060173, 0.0059418
7: -0.0110273, 0.0004206, -0.0109985, 0.0006102, -0.0116376, 0.0114191
8: -0.0112769, 0.0001720, -0.0110474, 0.0005026, -0.0117795, 0.0112194
9: 0.9810164, 0.9977967, 0.9810389, 0.9984642, -0.0174478, 0.0167577

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116651, upper bound: 0.0105503
time: 1.00 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116235, upper bound: 0.0100356
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0051233, 0.0059621, -0.0049535, 0.0061089, -0.0107518, 0.0104207
1: 0.0019534, 0.0117993, 0.0014964, 0.0115652, -0.0096118, 0.0103029
2: 0.0147073, 0.0316157, 0.0146978, 0.0316056, -0.0141401, 0.0141207
3: -0.0100924, -0.0025114, -0.0102569, -0.0027109, -0.0073814, 0.0077454
4: -0.0020468, 0.0024642, -0.0020137, 0.0028410, -0.0048877, 0.0044779
5: -0.0037130, 0.0067135, -0.0041455, 0.0065294, -0.0102423, 0.0108590
6: -0.0068169, -0.0009576, -0.0070853, -0.0010196, -0.0057973, 0.0061277
7: -0.0110273, 0.0004206, -0.0109330, 0.0009566, -0.0119840, 0.0113536
8: -0.0112769, 0.0001720, -0.0114510, 0.0000114, -0.0112883, 0.0116230
9: 0.9810164, 0.9977967, 0.9810869, 0.9987879, -0.0177715, 0.0167097

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120155, upper bound: 0.0101483
time: 1.04 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116235, upper bound: 0.0100356
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0054090, 0.0055572, -0.0102532, 0.0109456
1: 0.0019422, 0.0118258, 0.0019117, 0.0122140, -0.0102718, 0.0099140
2: 0.0146924, 0.0316475, 0.0142091, 0.0308843, -0.0135718, 0.0148333
3: -0.0101021, -0.0024892, -0.0099003, -0.0022676, -0.0078345, 0.0074111
4: -0.0020490, 0.0024703, -0.0021989, 0.0026808, -0.0047299, 0.0046692
5: -0.0037235, 0.0067224, -0.0038244, 0.0071021, -0.0108256, 0.0105468
6: -0.0068225, -0.0009472, -0.0068822, -0.0007798, -0.0060427, 0.0059350
7: -0.0110654, 0.0004304, -0.0110541, 0.0005229, -0.0115883, 0.0114845
8: -0.0112907, 0.0001849, -0.0110505, 0.0005376, -0.0118283, 0.0112354
9: 0.9809808, 0.9978092, 0.9810173, 0.9984320, -0.0174512, 0.0167919

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122473, upper bound: 0.0112211
time: 0.99 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122473, upper bound: 0.0117134
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0049977, 0.0061012, -0.0107552, 0.0104852
1: 0.0019422, 0.0118258, 0.0015312, 0.0116394, -0.0096972, 0.0102946
2: 0.0146924, 0.0316475, 0.0146494, 0.0316186, -0.0141863, 0.0142619
3: -0.0101021, -0.0024892, -0.0102366, -0.0026403, -0.0074619, 0.0077474
4: -0.0020490, 0.0024703, -0.0020201, 0.0028387, -0.0048878, 0.0044904
5: -0.0037235, 0.0067224, -0.0041516, 0.0065534, -0.0102769, 0.0108740
6: -0.0068225, -0.0009472, -0.0070756, -0.0009956, -0.0058269, 0.0061285
7: -0.0110654, 0.0004304, -0.0110044, 0.0009051, -0.0119705, 0.0114349
8: -0.0112907, 0.0001849, -0.0114595, 0.0000510, -0.0113418, 0.0116444
9: 0.9809808, 0.9978092, 0.9810556, 0.9987643, -0.0177836, 0.0167536

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122473, upper bound: 0.0112211
time: 1.04 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122473, upper bound: 0.0117134
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0055086, 0.0054643, -0.0053701, 0.0055777, -0.0106777, 0.0104124
1: 0.0022620, 0.0123909, 0.0018563, 0.0121564, -0.0098945, 0.0105346
2: 0.0142852, 0.0309471, 0.0142526, 0.0308812, -0.0140321, 0.0140537
3: -0.0097854, -0.0021454, -0.0099363, -0.0023272, -0.0074582, 0.0077910
4: -0.0022423, 0.0023532, -0.0021921, 0.0026863, -0.0049286, 0.0045453
5: -0.0034578, 0.0072499, -0.0038225, 0.0070810, -0.0105388, 0.0110724
6: -0.0066674, -0.0007199, -0.0068995, -0.0007996, -0.0058677, 0.0061795
7: -0.0111346, 0.0001052, -0.0109985, 0.0006102, -0.0117448, 0.0111037
8: -0.0109085, 0.0006345, -0.0110474, 0.0005026, -0.0114110, 0.0116819
9: 0.9809152, 0.9975743, 0.9810389, 0.9984642, -0.0175490, 0.0165354

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129925, upper bound: 0.0121185
time: 1.35 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129925, upper bound: 0.0121528
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055222, 0.0054834, -0.0054090, 0.0055572, -0.0106687, 0.0104744
1: 0.0022505, 0.0124151, 0.0019117, 0.0122140, -0.0099635, 0.0105034
2: 0.0142708, 0.0309785, 0.0142091, 0.0308843, -0.0140629, 0.0142191
3: -0.0097951, -0.0021241, -0.0099003, -0.0022676, -0.0075275, 0.0077761
4: -0.0022445, 0.0023594, -0.0021989, 0.0026808, -0.0049254, 0.0045583
5: -0.0034685, 0.0072581, -0.0038244, 0.0071021, -0.0105707, 0.0110825
6: -0.0066730, -0.0007098, -0.0068822, -0.0007798, -0.0058932, 0.0061724
7: -0.0111719, 0.0001152, -0.0110541, 0.0005229, -0.0116948, 0.0111693
8: -0.0109222, 0.0006468, -0.0110505, 0.0005376, -0.0114598, 0.0116973
9: 0.9808817, 0.9975876, 0.9810173, 0.9984320, -0.0175503, 0.0165703

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130843, upper bound: 0.0121247
time: 1.23 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130843, upper bound: 0.0122061
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054446, 0.0054694, -0.0050188, 0.0061196, -0.0111463, 0.0100378
1: 0.0022263, 0.0122826, 0.0015184, 0.0116790, -0.0094526, 0.0107642
2: 0.0143523, 0.0309202, 0.0146289, 0.0316473, -0.0147114, 0.0136299
3: -0.0098086, -0.0022434, -0.0102465, -0.0026087, -0.0071999, 0.0080031
4: -0.0022343, 0.0023523, -0.0020214, 0.0028451, -0.0050795, 0.0043737
5: -0.0034444, 0.0072175, -0.0041634, 0.0065634, -0.0100078, 0.0113810
6: -0.0066749, -0.0007536, -0.0070819, -0.0009847, -0.0056902, 0.0063283
7: -0.0110263, 0.0001705, -0.0110469, 0.0009164, -0.0119427, 0.0112174
8: -0.0108932, 0.0005799, -0.0114731, 0.0000679, -0.0109612, 0.0120530
9: 0.9809682, 0.9975894, 0.9810241, 0.9987798, -0.0178116, 0.0165653

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127123, upper bound: 0.0108446
time: 1.10 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123765, upper bound: 0.0107453
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054858, 0.0054465, -0.0050327, 0.0061389, -0.0112061, 0.0100271
1: 0.0022751, 0.0123450, 0.0015071, 0.0117046, -0.0094295, 0.0108379
2: 0.0143075, 0.0309194, 0.0146140, 0.0316795, -0.0148358, 0.0136640
3: -0.0097758, -0.0021798, -0.0102562, -0.0025867, -0.0071891, 0.0080764
4: -0.0022410, 0.0023465, -0.0020237, 0.0028512, -0.0050921, 0.0043702
5: -0.0034458, 0.0072394, -0.0041739, 0.0065721, -0.0100179, 0.0114132
6: -0.0066610, -0.0007330, -0.0070874, -0.0009741, -0.0056869, 0.0063544
7: -0.0110861, 0.0000939, -0.0110848, 0.0009264, -0.0120125, 0.0111788
8: -0.0108951, 0.0006155, -0.0114869, 0.0000807, -0.0109758, 0.0121025
9: 0.9809476, 0.9975584, 0.9809882, 0.9987926, -0.0178451, 0.0165702

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126763, upper bound: 0.0112146
time: 1.29 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126763, upper bound: 0.0117537
time: 2.36 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0050960, 0.0060071, -0.0053701, 0.0055777, -0.0102511, 0.0109437
1: 0.0018868, 0.0118011, 0.0018563, 0.0121564, -0.0102696, 0.0099448
2: 0.0147222, 0.0316817, 0.0142526, 0.0308812, -0.0135542, 0.0147813
3: -0.0101218, -0.0025238, -0.0099363, -0.0023272, -0.0077946, 0.0074125
4: -0.0020521, 0.0025187, -0.0021921, 0.0026863, -0.0047384, 0.0047108
5: -0.0037964, 0.0066987, -0.0038225, 0.0070810, -0.0108774, 0.0105212
6: -0.0068614, -0.0009460, -0.0068995, -0.0007996, -0.0060618, 0.0059534
7: -0.0110757, 0.0004826, -0.0109985, 0.0006102, -0.0116859, 0.0114810
8: -0.0113195, 0.0001481, -0.0110474, 0.0005026, -0.0118221, 0.0111955
9: 0.9809585, 0.9979365, 0.9810389, 0.9984642, -0.0175058, 0.0168976

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116436, upper bound: 0.0106070
time: 0.94 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116021, upper bound: 0.0100712
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0050960, 0.0060071, -0.0049535, 0.0061089, -0.0107436, 0.0104754
1: 0.0018868, 0.0118011, 0.0014964, 0.0115652, -0.0096784, 0.0103047
2: 0.0147222, 0.0316817, 0.0146978, 0.0316056, -0.0141593, 0.0141921
3: -0.0101218, -0.0025238, -0.0102569, -0.0027109, -0.0074109, 0.0077331
4: -0.0020521, 0.0025187, -0.0020137, 0.0028410, -0.0048931, 0.0045323
5: -0.0037964, 0.0066987, -0.0041455, 0.0065294, -0.0103258, 0.0108442
6: -0.0068614, -0.0009460, -0.0070853, -0.0010196, -0.0058418, 0.0061393
7: -0.0110757, 0.0004826, -0.0109330, 0.0009566, -0.0120323, 0.0114155
8: -0.0113195, 0.0001481, -0.0114510, 0.0000114, -0.0113309, 0.0115991
9: 0.9809585, 0.9979365, 0.9810869, 0.9987879, -0.0178294, 0.0168496

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116436, upper bound: 0.0106070
time: 0.96 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116021, upper bound: 0.0100712
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0054090, 0.0055572, -0.0102434, 0.0110004
1: 0.0018756, 0.0118278, 0.0019117, 0.0122140, -0.0103384, 0.0099161
2: 0.0147073, 0.0317128, 0.0142091, 0.0308843, -0.0135899, 0.0149048
3: -0.0101313, -0.0025013, -0.0099003, -0.0022676, -0.0078637, 0.0073990
4: -0.0020544, 0.0025248, -0.0021989, 0.0026808, -0.0047352, 0.0047237
5: -0.0038068, 0.0067076, -0.0038244, 0.0071021, -0.0109089, 0.0105320
6: -0.0068669, -0.0009356, -0.0068822, -0.0007798, -0.0060871, 0.0059466
7: -0.0111134, 0.0004922, -0.0110541, 0.0005229, -0.0116363, 0.0115463
8: -0.0113331, 0.0001610, -0.0110505, 0.0005376, -0.0118707, 0.0112115
9: 0.9809233, 0.9979489, 0.9810173, 0.9984320, -0.0175087, 0.0169316

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122225, upper bound: 0.0112609
time: 1.07 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122225, upper bound: 0.0117457
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0049977, 0.0061012, -0.0107472, 0.0105389
1: 0.0018756, 0.0118278, 0.0015312, 0.0116394, -0.0097637, 0.0102966
2: 0.0147073, 0.0317128, 0.0146494, 0.0316186, -0.0142054, 0.0143321
3: -0.0101313, -0.0025013, -0.0102366, -0.0026403, -0.0074911, 0.0077353
4: -0.0020544, 0.0025248, -0.0020201, 0.0028387, -0.0048931, 0.0045449
5: -0.0038068, 0.0067076, -0.0041516, 0.0065534, -0.0103602, 0.0108592
6: -0.0068669, -0.0009356, -0.0070756, -0.0009956, -0.0058713, 0.0061400
7: -0.0111134, 0.0004922, -0.0110044, 0.0009051, -0.0120185, 0.0114967
8: -0.0113331, 0.0001610, -0.0114595, 0.0000510, -0.0113841, 0.0116205
9: 0.9809233, 0.9979489, 0.9810556, 0.9987643, -0.0178410, 0.0168933

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122225, upper bound: 0.0112609
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122225, upper bound: 0.0117457
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054756, 0.0054251, -0.0055086, 0.0054643, -0.0104842, 0.0104807
1: 0.0022835, 0.0122868, 0.0022620, 0.0123909, -0.0101074, 0.0100248
2: 0.0143318, 0.0308501, 0.0142852, 0.0309471, -0.0139911, 0.0139906
3: -0.0097814, -0.0022327, -0.0097854, -0.0021454, -0.0076361, 0.0075527
4: -0.0022229, 0.0022993, -0.0022423, 0.0023532, -0.0045761, 0.0045416
5: -0.0033758, 0.0072324, -0.0034578, 0.0072499, -0.0106258, 0.0106903
6: -0.0066369, -0.0007657, -0.0066674, -0.0007199, -0.0059170, 0.0059017
7: -0.0109823, 0.0001146, -0.0111346, 0.0001052, -0.0110875, 0.0112491
8: -0.0108561, 0.0006082, -0.0109085, 0.0006345, -0.0114906, 0.0115167
9: 0.9810299, 0.9974655, 0.9809152, 0.9975743, -0.0165444, 0.0165503

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119038, upper bound: 0.0114900
time: 1.07 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114825, upper bound: 0.0114414
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055163, 0.0054050, -0.0055222, 0.0054834, -0.0105470, 0.0104718
1: 0.0023342, 0.0123476, 0.0022505, 0.0124151, -0.0100809, 0.0100971
2: 0.0142882, 0.0308556, 0.0142708, 0.0309785, -0.0141576, 0.0140253
3: -0.0097493, -0.0021712, -0.0097951, -0.0021241, -0.0076251, 0.0076239
4: -0.0022296, 0.0022939, -0.0022445, 0.0023594, -0.0045890, 0.0045385
5: -0.0033783, 0.0072546, -0.0034685, 0.0072581, -0.0106365, 0.0107231
6: -0.0066207, -0.0007445, -0.0066730, -0.0007098, -0.0059109, 0.0059285
7: -0.0110440, 0.0000358, -0.0111719, 0.0001152, -0.0111591, 0.0112077
8: -0.0108593, 0.0006442, -0.0109222, 0.0006468, -0.0115061, 0.0115664
9: 0.9810047, 0.9974326, 0.9808817, 0.9975876, -0.0165830, 0.0165509

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122226, upper bound: 0.0122120
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122226, upper bound: 0.0122748
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054756, 0.0054251, -0.0050960, 0.0060071, -0.0110155, 0.0100541
1: 0.0022835, 0.0122868, 0.0018868, 0.0118011, -0.0095177, 0.0104000
2: 0.0143318, 0.0308501, 0.0147222, 0.0316817, -0.0147187, 0.0135127
3: -0.0097814, -0.0022327, -0.0101218, -0.0025238, -0.0072576, 0.0078891
4: -0.0022229, 0.0022993, -0.0020521, 0.0025187, -0.0047416, 0.0043514
5: -0.0033758, 0.0072324, -0.0037964, 0.0066987, -0.0100745, 0.0110289
6: -0.0066369, -0.0007657, -0.0068614, -0.0009460, -0.0056909, 0.0060958
7: -0.0109823, 0.0001146, -0.0110757, 0.0004826, -0.0114648, 0.0111902
8: -0.0108561, 0.0006082, -0.0113195, 0.0001481, -0.0110042, 0.0119278
9: 0.9810299, 0.9974655, 0.9809585, 0.9979365, -0.0169066, 0.0165070

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114817, upper bound: 0.0107781
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109613, upper bound: 0.0106335
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055163, 0.0054050, -0.0051101, 0.0060261, -0.0110729, 0.0100466
1: 0.0023342, 0.0123476, 0.0018756, 0.0118278, -0.0094936, 0.0104720
2: 0.0142882, 0.0308556, 0.0147073, 0.0317128, -0.0148433, 0.0135523
3: -0.0097493, -0.0021712, -0.0101313, -0.0025013, -0.0072479, 0.0079601
4: -0.0022296, 0.0022939, -0.0020544, 0.0025248, -0.0047544, 0.0043483
5: -0.0033783, 0.0072546, -0.0038068, 0.0067076, -0.0100859, 0.0110614
6: -0.0066207, -0.0007445, -0.0068669, -0.0009356, -0.0056851, 0.0061224
7: -0.0110440, 0.0000358, -0.0111134, 0.0004922, -0.0115362, 0.0111492
8: -0.0108593, 0.0006442, -0.0113331, 0.0001610, -0.0110203, 0.0119773
9: 0.9810047, 0.9974326, 0.9809233, 0.9979489, -0.0169442, 0.0165093

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116337, upper bound: 0.0111805
time: 0.99 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116337, upper bound: 0.0117339
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0054858, 0.0054465, -0.0101069, 0.0110048
1: 0.0019422, 0.0118258, 0.0022751, 0.0123450, -0.0104028, 0.0095507
2: 0.0146924, 0.0316475, 0.0143075, 0.0309194, -0.0136034, 0.0147892
3: -0.0101021, -0.0024892, -0.0097758, -0.0021798, -0.0079223, 0.0072866
4: -0.0020490, 0.0024703, -0.0022410, 0.0023465, -0.0043955, 0.0047113
5: -0.0037235, 0.0067224, -0.0034458, 0.0072394, -0.0109628, 0.0101682
6: -0.0068225, -0.0009472, -0.0066610, -0.0007330, -0.0060895, 0.0057138
7: -0.0110654, 0.0004304, -0.0110861, 0.0000939, -0.0111593, 0.0115165
8: -0.0112907, 0.0001849, -0.0108951, 0.0006155, -0.0119062, 0.0110799
9: 0.9809808, 0.9978092, 0.9809476, 0.9975584, -0.0165776, 0.0168617

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110168, upper bound: 0.0112186
time: 1.03 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0110168, upper bound: 0.0117209
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0051373, 0.0059815, -0.0050755, 0.0059886, -0.0106115, 0.0105552
1: 0.0019422, 0.0118258, 0.0019000, 0.0117650, -0.0098229, 0.0099257
2: 0.0146924, 0.0316475, 0.0147426, 0.0316538, -0.0142141, 0.0142186
3: -0.0101021, -0.0024892, -0.0101119, -0.0025534, -0.0075487, 0.0076227
4: -0.0020490, 0.0024703, -0.0020509, 0.0025120, -0.0045611, 0.0045212
5: -0.0037235, 0.0067224, -0.0037841, 0.0066885, -0.0104119, 0.0105065
6: -0.0068225, -0.0009472, -0.0068549, -0.0009567, -0.0058658, 0.0059077
7: -0.0110654, 0.0004304, -0.0110354, 0.0004710, -0.0115364, 0.0114659
8: -0.0112907, 0.0001849, -0.0113057, 0.0001315, -0.0114223, 0.0114906
9: 0.9809808, 0.9978092, 0.9809884, 0.9979210, -0.0169403, 0.0168208

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110168, upper bound: 0.0112186
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0110168, upper bound: 0.0117209
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054446, 0.0054694, -0.0055086, 0.0054643, -0.0104920, 0.0105535
1: 0.0022263, 0.0122826, 0.0022620, 0.0123909, -0.0101645, 0.0100207
2: 0.0143523, 0.0309202, 0.0142852, 0.0309471, -0.0141449, 0.0141937
3: -0.0098086, -0.0022434, -0.0097854, -0.0021454, -0.0076633, 0.0075420
4: -0.0022343, 0.0023523, -0.0022423, 0.0023532, -0.0045875, 0.0045947
5: -0.0034444, 0.0072175, -0.0034578, 0.0072499, -0.0106943, 0.0106754
6: -0.0066749, -0.0007536, -0.0066674, -0.0007199, -0.0059549, 0.0059138
7: -0.0110263, 0.0001705, -0.0111346, 0.0001052, -0.0111315, 0.0113051
8: -0.0108932, 0.0005799, -0.0109085, 0.0006345, -0.0115277, 0.0114883
9: 0.9809682, 0.9975894, 0.9809152, 0.9975743, -0.0166062, 0.0166742

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121688, upper bound: 0.0122118
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121688, upper bound: 0.0122226
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0054858, 0.0054465, -0.0055222, 0.0054834, -0.0105547, 0.0105440
1: 0.0022751, 0.0123450, 0.0022505, 0.0124151, -0.0101401, 0.0100946
2: 0.0143075, 0.0309194, 0.0142708, 0.0309785, -0.0142996, 0.0142278
3: -0.0097758, -0.0021798, -0.0097951, -0.0021241, -0.0076516, 0.0076152
4: -0.0022410, 0.0023465, -0.0022445, 0.0023594, -0.0046004, 0.0045910
5: -0.0034458, 0.0072394, -0.0034685, 0.0072581, -0.0107039, 0.0107079
6: -0.0066610, -0.0007330, -0.0066730, -0.0007098, -0.0059512, 0.0059400
7: -0.0110861, 0.0000939, -0.0111719, 0.0001152, -0.0112012, 0.0112658
8: -0.0108951, 0.0006155, -0.0109222, 0.0006468, -0.0115419, 0.0115377
9: 0.9809476, 0.9975584, 0.9808817, 0.9975876, -0.0166401, 0.0166767

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121738, upper bound: 0.0122520
time: 1.16 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121738, upper bound: 0.0123212
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054446, 0.0054694, -0.0050960, 0.0060071, -0.0110258, 0.0101264
1: 0.0022263, 0.0122826, 0.0018868, 0.0118011, -0.0095748, 0.0103958
2: 0.0143523, 0.0309202, 0.0147222, 0.0316817, -0.0148794, 0.0137143
3: -0.0098086, -0.0022434, -0.0101218, -0.0025238, -0.0072848, 0.0078784
4: -0.0022343, 0.0023523, -0.0020521, 0.0025187, -0.0047530, 0.0044045
5: -0.0034444, 0.0072175, -0.0037964, 0.0066987, -0.0101431, 0.0110139
6: -0.0066749, -0.0007536, -0.0068614, -0.0009460, -0.0057288, 0.0061078
7: -0.0110263, 0.0001705, -0.0110757, 0.0004826, -0.0115088, 0.0112462
8: -0.0108932, 0.0005799, -0.0113195, 0.0001481, -0.0110413, 0.0118994
9: 0.9809682, 0.9975894, 0.9809585, 0.9979365, -0.0169683, 0.0166309

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114529, upper bound: 0.0108499
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109883, upper bound: 0.0107486
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054858, 0.0054465, -0.0051101, 0.0060261, -0.0110833, 0.0101183
1: 0.0022751, 0.0123450, 0.0018756, 0.0118278, -0.0095527, 0.0104694
2: 0.0143075, 0.0309194, 0.0147073, 0.0317128, -0.0149928, 0.0137539
3: -0.0097758, -0.0021798, -0.0101313, -0.0025013, -0.0072744, 0.0079515
4: -0.0022410, 0.0023465, -0.0020544, 0.0025248, -0.0047658, 0.0044009
5: -0.0034458, 0.0072394, -0.0038068, 0.0067076, -0.0101534, 0.0110461
6: -0.0066610, -0.0007330, -0.0068669, -0.0009356, -0.0057254, 0.0061339
7: -0.0110861, 0.0000939, -0.0111134, 0.0004922, -0.0115783, 0.0112073
8: -0.0108951, 0.0006155, -0.0113331, 0.0001610, -0.0110560, 0.0119486
9: 0.9809476, 0.9975584, 0.9809233, 0.9979489, -0.0170013, 0.0166351

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115866, upper bound: 0.0112146
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115866, upper bound: 0.0117643
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0054858, 0.0054465, -0.0101183, 0.0110833
1: 0.0018756, 0.0118278, 0.0022751, 0.0123450, -0.0104694, 0.0095527
2: 0.0147073, 0.0317128, 0.0143075, 0.0309194, -0.0137539, 0.0149928
3: -0.0101313, -0.0025013, -0.0097758, -0.0021798, -0.0079515, 0.0072744
4: -0.0020544, 0.0025248, -0.0022410, 0.0023465, -0.0044009, 0.0047658
5: -0.0038068, 0.0067076, -0.0034458, 0.0072394, -0.0110461, 0.0101534
6: -0.0068669, -0.0009356, -0.0066610, -0.0007330, -0.0061339, 0.0057254
7: -0.0111134, 0.0004922, -0.0110861, 0.0000939, -0.0112073, 0.0115783
8: -0.0113331, 0.0001610, -0.0108951, 0.0006155, -0.0119486, 0.0110560
9: 0.9809233, 0.9979489, 0.9809476, 0.9975584, -0.0166351, 0.0170013

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109789, upper bound: 0.0112609
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0109789, upper bound: 0.0117554
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0051101, 0.0060261, -0.0050755, 0.0059886, -0.0106251, 0.0106297
1: 0.0018756, 0.0118278, 0.0019000, 0.0117650, -0.0098894, 0.0099278
2: 0.0147073, 0.0317128, 0.0147426, 0.0316538, -0.0143560, 0.0144080
3: -0.0101313, -0.0025013, -0.0101119, -0.0025534, -0.0075779, 0.0076105
4: -0.0020544, 0.0025248, -0.0020509, 0.0025120, -0.0045664, 0.0045757
5: -0.0038068, 0.0067076, -0.0037841, 0.0066885, -0.0104953, 0.0104917
6: -0.0068669, -0.0009356, -0.0068549, -0.0009567, -0.0059102, 0.0059193
7: -0.0111134, 0.0004922, -0.0110354, 0.0004710, -0.0115844, 0.0115276
8: -0.0113331, 0.0001610, -0.0113057, 0.0001315, -0.0114646, 0.0114667
9: 0.9809233, 0.9979489, 0.9809884, 0.9979210, -0.0169978, 0.0169605

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 221

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109789, upper bound: 0.0112609
time: 1.04 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0109789, upper bound: 0.0117554
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0055394, 0.0054222, -0.0053429, 0.0056248, -0.0107463, 0.0103545
1: 0.0023217, 0.0123942, 0.0017954, 0.0121674, -0.0098457, 0.0105988
2: 0.0142653, 0.0308822, 0.0142731, 0.0309513, -0.0140893, 0.0140038
3: -0.0097585, -0.0021359, -0.0099633, -0.0023389, -0.0074196, 0.0078273
4: -0.0022310, 0.0023002, -0.0022085, 0.0027481, -0.0049791, 0.0045087
5: -0.0033898, 0.0072652, -0.0039041, 0.0070808, -0.0104706, 0.0111693
6: -0.0066267, -0.0007314, -0.0069417, -0.0007808, -0.0058458, 0.0062102
7: -0.0110926, 0.0000465, -0.0110484, 0.0006698, -0.0117624, 0.0110949
8: -0.0108721, 0.0006634, -0.0110875, 0.0004776, -0.0113497, 0.0117508
9: 0.9809726, 0.9974478, 0.9809657, 0.9986097, -0.0176370, 0.0164821

Time for backsubstitution: 1.63 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.34 + 596.95 = 600.29 seconds
