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
execution time: IAR + RelationalAnalysis = 1.28 + 1.88 = 3.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0137566, upper bound: 0.0137566

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135920, upper bound: 0.0125540
time: 1.17 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135922, upper bound: 0.0135922
time: 1.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 9, lower bound: -0.0135920, upper bound: 0.0125540
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.41
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

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0125540
time: 1.33 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0125540
time: 1.22 seconds

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

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135920
time: 1.62 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135922
time: 1.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.93 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0125540
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0125540
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135920
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 9, lower bound: -0.0125540, upper bound: 0.0135922

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056460, -0.0057254, 0.0056460, -0.0109882, 0.0109882
1: 0.0020836, 0.0126449, 0.0020836, 0.0126449, -0.0105613, 0.0105613
2: 0.0140275, 0.0311463, 0.0140275, 0.0311463, -0.0148571, 0.0148571
3: -0.0098849, -0.0019333, -0.0098849, -0.0019333, -0.0079516, 0.0079516
4: -0.0023122, 0.0024672, -0.0023122, 0.0024672, -0.0047793, 0.0047793
5: -0.0036777, 0.0074717, -0.0036777, 0.0074717, -0.0111494, 0.0111494
6: -0.0067717, -0.0005886, -0.0067717, -0.0005886, -0.0061831, 0.0061831
7: -0.0114247, 0.0002549, -0.0114247, 0.0002549, -0.0116796, 0.0116796
8: -0.0110869, 0.0008870, -0.0110869, 0.0008870, -0.0119739, 0.0119739
9: 0.9805632, 0.9978729, 0.9805632, 0.9978729, -0.0173097, 0.0173097

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123442
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
time: 1.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057254, 0.0056460, -0.0056207, 0.0058012, -0.0111854, 0.0109174
1: 0.0020836, 0.0126449, 0.0016581, 0.0125317, -0.0104481, 0.0109868
2: 0.0140275, 0.0311463, 0.0139490, 0.0311798, -0.0148952, 0.0149188
3: -0.0098849, -0.0019333, -0.0100363, -0.0020369, -0.0078479, 0.0081031
4: -0.0023122, 0.0024672, -0.0022889, 0.0028598, -0.0051720, 0.0047561
5: -0.0036777, 0.0074717, -0.0041294, 0.0073334, -0.0110111, 0.0116011
6: -0.0067717, -0.0005886, -0.0070354, -0.0006148, -0.0061569, 0.0064468
7: -0.0114247, 0.0002549, -0.0114478, 0.0007463, -0.0121710, 0.0117027
8: -0.0110869, 0.0008870, -0.0112797, 0.0007809, -0.0118678, 0.0121667
9: 0.9805632, 0.9978729, 0.9805573, 0.9988881, -0.0183249, 0.0173156

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123442
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
time: 1.12 seconds

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

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132899
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134335
time: 1.31 seconds

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

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132910
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134342
time: 4.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.62 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123442
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123442
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0124884, upper bound: 0.0123738
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132899
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134335
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0132910
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.62
Output dim: 9, lower bound: -0.0123738, upper bound: 0.0134342

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0057254, 0.0056215, -0.0109343, 0.0107583
1: 0.0022875, 0.0126435, 0.0021106, 0.0126441, -0.0103566, 0.0105329
2: 0.0140086, 0.0309421, 0.0140277, 0.0311200, -0.0146542, 0.0145179
3: -0.0097759, -0.0019237, -0.0098707, -0.0019334, -0.0078425, 0.0079470
4: -0.0022995, 0.0023202, -0.0023121, 0.0024479, -0.0047474, 0.0046323
5: -0.0034181, 0.0074809, -0.0036436, 0.0074710, -0.0108891, 0.0111246
6: -0.0066403, -0.0006086, -0.0067542, -0.0005905, -0.0060499, 0.0061456
7: -0.0113376, 0.0000733, -0.0114159, 0.0002310, -0.0115686, 0.0114892
8: -0.0108967, 0.0009173, -0.0110619, 0.0008865, -0.0117832, 0.0119792
9: 0.9806842, 0.9975065, 0.9805752, 0.9978237, -0.0171396, 0.0169313

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114166, upper bound: 0.0117004
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0112916
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0057254, 0.0056364, -0.0109608, 0.0108114
1: 0.0022284, 0.0126409, 0.0020930, 0.0126446, -0.0104163, 0.0105479
2: 0.0140281, 0.0310068, 0.0140275, 0.0311371, -0.0148201, 0.0145948
3: -0.0098028, -0.0019340, -0.0098795, -0.0019333, -0.0078695, 0.0079456
4: -0.0023119, 0.0023732, -0.0023121, 0.0024610, -0.0047729, 0.0046854
5: -0.0034860, 0.0074682, -0.0036653, 0.0074715, -0.0109574, 0.0111335
6: -0.0066810, -0.0005981, -0.0067658, -0.0005892, -0.0060919, 0.0061677
7: -0.0113801, 0.0001318, -0.0114218, 0.0002469, -0.0116270, 0.0115537
8: -0.0109331, 0.0008848, -0.0110769, 0.0008868, -0.0118199, 0.0119617
9: 0.9806254, 0.9976319, 0.9805671, 0.9978570, -0.0172316, 0.0170648

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0124884
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0124884
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0056206, 0.0057761, -0.0111311, 0.0106875
1: 0.0022875, 0.0126435, 0.0016855, 0.0125308, -0.0102434, 0.0109579
2: 0.0140086, 0.0309421, 0.0139492, 0.0311536, -0.0146921, 0.0145796
3: -0.0097759, -0.0019237, -0.0100221, -0.0020372, -0.0077387, 0.0080984
4: -0.0022995, 0.0023202, -0.0022889, 0.0028398, -0.0051393, 0.0046091
5: -0.0034181, 0.0074809, -0.0040944, 0.0073327, -0.0107508, 0.0115753
6: -0.0066403, -0.0006086, -0.0070177, -0.0006168, -0.0060235, 0.0064091
7: -0.0113376, 0.0000733, -0.0114388, 0.0007220, -0.0120596, 0.0115121
8: -0.0108967, 0.0009173, -0.0112546, 0.0007804, -0.0116771, 0.0121718
9: 0.9806842, 0.9975065, 0.9805698, 0.9988371, -0.0181530, 0.0169367

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126887, upper bound: 0.0116882
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126742, upper bound: 0.0112916
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057253, 0.0054984, -0.0056206, 0.0057917, -0.0111598, 0.0107405
1: 0.0022284, 0.0126409, 0.0016672, 0.0125314, -0.0103031, 0.0109736
2: 0.0140281, 0.0310068, 0.0139491, 0.0311707, -0.0148572, 0.0146565
3: -0.0098028, -0.0019340, -0.0100310, -0.0020370, -0.0077658, 0.0080970
4: -0.0023119, 0.0023732, -0.0022889, 0.0028539, -0.0051659, 0.0046621
5: -0.0034860, 0.0074682, -0.0041174, 0.0073332, -0.0108192, 0.0115856
6: -0.0066810, -0.0005981, -0.0070295, -0.0006155, -0.0060656, 0.0064314
7: -0.0113801, 0.0001318, -0.0114448, 0.0007385, -0.0121186, 0.0115767
8: -0.0109331, 0.0008848, -0.0112697, 0.0007808, -0.0117138, 0.0121545
9: 0.9806254, 0.9976319, 0.9805615, 0.9988730, -0.0182476, 0.0170704

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132899, upper bound: 0.0123738
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132899, upper bound: 0.0123738
time: 1.18 seconds

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

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114141, upper bound: 0.0128118
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125193
time: 1.02 seconds

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

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
time: 1.40 seconds

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

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118678, upper bound: 0.0128455
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119171, upper bound: 0.0126266
time: 0.98 seconds

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

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342
time: 1.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.89 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0114166, upper bound: 0.0117004
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0112916
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0124884
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0124416, upper bound: 0.0124884
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0126887, upper bound: 0.0116882
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0126742, upper bound: 0.0112916
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0132899, upper bound: 0.0123738
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0132899, upper bound: 0.0123738
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0114141, upper bound: 0.0128118
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125193
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0123442, upper bound: 0.0134335
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0118678, upper bound: 0.0128455
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0119171, upper bound: 0.0126266
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 9, lower bound: -0.0124584, upper bound: 0.0134342

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0057227, 0.0053017, -0.0106200, 0.0107544
1: 0.0022875, 0.0126435, 0.0023583, 0.0126287, -0.0103412, 0.0102852
2: 0.0140086, 0.0309421, 0.0140321, 0.0306935, -0.0142215, 0.0145063
3: -0.0097759, -0.0019237, -0.0096967, -0.0019372, -0.0078387, 0.0077730
4: -0.0022995, 0.0023202, -0.0023097, 0.0023047, -0.0046042, 0.0046298
5: -0.0034181, 0.0074809, -0.0033384, 0.0074571, -0.0108752, 0.0108193
6: -0.0066403, -0.0006086, -0.0066049, -0.0006216, -0.0060187, 0.0059963
7: -0.0113376, 0.0000733, -0.0112777, 0.0000192, -0.0113569, 0.0113510
8: -0.0108967, 0.0009173, -0.0107755, 0.0008776, -0.0117743, 0.0116928
9: 0.9806842, 0.9975065, 0.9807646, 0.9974712, -0.0167870, 0.0167419

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0112916
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0112916
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B1

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

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117004, upper bound: 0.0114166
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2

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

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117004, upper bound: 0.0114166
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057575, 0.0054564, -0.0056173, 0.0054546, -0.0108173, 0.0106829
1: 0.0022875, 0.0126435, 0.0019295, 0.0125133, -0.0102258, 0.0107140
2: 0.0140086, 0.0309421, 0.0139541, 0.0307275, -0.0142602, 0.0145674
3: -0.0097759, -0.0019237, -0.0098491, -0.0020419, -0.0077340, 0.0079254
4: -0.0022995, 0.0023202, -0.0022857, 0.0026948, -0.0049943, 0.0046058
5: -0.0034181, 0.0074809, -0.0037913, 0.0073173, -0.0107354, 0.0112723
6: -0.0066403, -0.0006086, -0.0068709, -0.0006510, -0.0059893, 0.0062623
7: -0.0113376, 0.0000733, -0.0112952, 0.0005126, -0.0118503, 0.0113685
8: -0.0108967, 0.0009173, -0.0109686, 0.0007703, -0.0116670, 0.0118859
9: 0.9806842, 0.9975065, 0.9807639, 0.9984745, -0.0177903, 0.0167426

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126667, upper bound: 0.0112916
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126667, upper bound: 0.0112916
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057566, 0.0052978, -0.0067659, 0.0050782, -0.0105309, 0.0117107
1: 0.0024126, 0.0126361, 0.0021609, 0.0137127, -0.0113002, 0.0104752
2: 0.0140101, 0.0307215, 0.0125542, 0.0302091, -0.0142433, 0.0158364
3: -0.0096892, -0.0019252, -0.0096418, -0.0011639, -0.0085253, 0.0077166
4: -0.0022987, 0.0022469, -0.0026821, 0.0025270, -0.0048257, 0.0049290
5: -0.0032760, 0.0074742, -0.0034893, 0.0087315, -0.0120075, 0.0109635
6: -0.0065691, -0.0006251, -0.0067080, -0.0000424, -0.0065267, 0.0060830
7: -0.0112614, -0.0000402, -0.0118602, 0.0002460, -0.0115074, 0.0118200
8: -0.0107622, 0.0009133, -0.0106494, 0.0021933, -0.0129554, 0.0115628
9: 0.9807898, 0.9973302, 0.9798483, 0.9981731, -0.0173832, 0.0174819

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126742, upper bound: 0.0112916
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126742, upper bound: 0.0112916
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1

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

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128118, upper bound: 0.0114141
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125193, upper bound: 0.0113225
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2_B2

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

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128118, upper bound: 0.0114141
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125193, upper bound: 0.0113225
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0057227, 0.0053017, -0.0105663, 0.0109506
1: 0.0018636, 0.0125181, 0.0023583, 0.0126287, -0.0107651, 0.0101598
2: 0.0139272, 0.0309755, 0.0140321, 0.0306935, -0.0143586, 0.0145435
3: -0.0099271, -0.0020233, -0.0096967, -0.0019372, -0.0079899, 0.0076733
4: -0.0022731, 0.0027077, -0.0023097, 0.0023047, -0.0045777, 0.0050173
5: -0.0038639, 0.0073364, -0.0033384, 0.0074571, -0.0113210, 0.0106749
6: -0.0069024, -0.0006412, -0.0066049, -0.0006216, -0.0062807, 0.0059636
7: -0.0113537, 0.0005622, -0.0112777, 0.0000192, -0.0113729, 0.0118399
8: -0.0110881, 0.0008096, -0.0107755, 0.0008776, -0.0119657, 0.0115851
9: 0.9806883, 0.9985073, 0.9807646, 0.9974712, -0.0167828, 0.0177426

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125168
time: 2.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125193
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056480, 0.0054511, -0.0068767, 0.0049221, -0.0102554, 0.0119751
1: 0.0019878, 0.0125082, 0.0026041, 0.0138501, -0.0118623, 0.0099041
2: 0.0139311, 0.0307555, 0.0126336, 0.0301669, -0.0142841, 0.0158046
3: -0.0098412, -0.0020265, -0.0094892, -0.0010624, -0.0087787, 0.0074626
4: -0.0022716, 0.0026360, -0.0027127, 0.0021342, -0.0044058, 0.0053487
5: -0.0037250, 0.0073276, -0.0030180, 0.0088819, -0.0126069, 0.0103456
6: -0.0068301, -0.0006598, -0.0064352, -0.0000001, -0.0068300, 0.0057754
7: -0.0112750, 0.0004466, -0.0118616, -0.0002472, -0.0110277, 0.0123081
8: -0.0109526, 0.0008028, -0.0104579, 0.0023033, -0.0132559, 0.0112608
9: 0.9807976, 0.9983351, 0.9798118, 0.9971586, -0.0163609, 0.0185233

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125168
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125193
time: 0.94 seconds

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

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116882, upper bound: 0.0126887
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
time: 1.22 seconds

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

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116882, upper bound: 0.0126887
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056508, 0.0056098, -0.0056173, 0.0054546, -0.0108223, 0.0109250
1: 0.0018636, 0.0125181, 0.0019295, 0.0125133, -0.0106497, 0.0105886
2: 0.0139272, 0.0309755, 0.0139541, 0.0307275, -0.0147047, 0.0148209
3: -0.0099271, -0.0020233, -0.0098491, -0.0020419, -0.0078852, 0.0078257
4: -0.0022731, 0.0027077, -0.0022857, 0.0026948, -0.0049679, 0.0049934
5: -0.0038639, 0.0073364, -0.0037913, 0.0073173, -0.0111812, 0.0111278
6: -0.0069024, -0.0006412, -0.0068709, -0.0006510, -0.0062514, 0.0062296
7: -0.0113537, 0.0005622, -0.0112952, 0.0005126, -0.0118663, 0.0118573
8: -0.0110881, 0.0008096, -0.0109686, 0.0007703, -0.0118585, 0.0117781
9: 0.9806883, 0.9985073, 0.9807639, 0.9984745, -0.0177861, 0.0177433

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118666, upper bound: 0.0126129
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118666, upper bound: 0.0126266
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056480, 0.0054511, -0.0067659, 0.0050782, -0.0105010, 0.0119529
1: 0.0019878, 0.0125082, 0.0021609, 0.0137127, -0.0117250, 0.0103472
2: 0.0139311, 0.0307555, 0.0125542, 0.0302091, -0.0145236, 0.0160663
3: -0.0098412, -0.0020265, -0.0096418, -0.0011639, -0.0086773, 0.0076153
4: -0.0022716, 0.0026360, -0.0026821, 0.0025270, -0.0047986, 0.0053181
5: -0.0037250, 0.0073276, -0.0034893, 0.0087315, -0.0124566, 0.0108169
6: -0.0068301, -0.0006598, -0.0067080, -0.0000424, -0.0067877, 0.0060482
7: -0.0112750, 0.0004466, -0.0118602, 0.0002460, -0.0115210, 0.0123068
8: -0.0109526, 0.0008028, -0.0106494, 0.0021933, -0.0131459, 0.0114523
9: 0.9807976, 0.9983351, 0.9798483, 0.9981731, -0.0173754, 0.0184869

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119171, upper bound: 0.0126129
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119171, upper bound: 0.0126266
time: 0.99 seconds

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

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120662, upper bound: 0.0127615
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118585, upper bound: 0.0127808
time: 1.01 seconds

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

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120662, upper bound: 0.0127615
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118585, upper bound: 0.0127808
time: 1.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.69 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0112916
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0112916
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0117004, upper bound: 0.0114166
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0117004, upper bound: 0.0114166
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0126667, upper bound: 0.0112916
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0126667, upper bound: 0.0112916
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0126742, upper bound: 0.0112916
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0126742, upper bound: 0.0112916
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0128118, upper bound: 0.0114141
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0125193, upper bound: 0.0113225
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0128118, upper bound: 0.0114141
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0125193, upper bound: 0.0113225
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125168
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125193
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125168
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0113225, upper bound: 0.0125193
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0116882, upper bound: 0.0126887
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0116882, upper bound: 0.0126887
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0118666, upper bound: 0.0126129
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0118666, upper bound: 0.0126266
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0119171, upper bound: 0.0126129
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0119171, upper bound: 0.0126266
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0120662, upper bound: 0.0127615
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0118585, upper bound: 0.0127808
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0120662, upper bound: 0.0127615
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 9, lower bound: -0.0118585, upper bound: 0.0127808

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0057575, 0.0054564, -0.0107247, 0.0104732
1: 0.0024777, 0.0126253, 0.0022875, 0.0126435, -0.0101658, 0.0103379
2: 0.0140326, 0.0305792, 0.0140086, 0.0309421, -0.0143833, 0.0140111
3: -0.0096271, -0.0019377, -0.0097759, -0.0019237, -0.0077034, 0.0078381
4: -0.0023095, 0.0022292, -0.0022995, 0.0023202, -0.0046296, 0.0045287
5: -0.0031822, 0.0074543, -0.0034181, 0.0074809, -0.0106631, 0.0108725
6: -0.0065304, -0.0006294, -0.0066403, -0.0006086, -0.0059218, 0.0060110
7: -0.0112415, -0.0000816, -0.0113376, 0.0000733, -0.0113148, 0.0112560
8: -0.0106450, 0.0008758, -0.0108967, 0.0009173, -0.0115623, 0.0117725
9: 0.9808148, 0.9972770, 0.9806842, 0.9975065, -0.0166917, 0.0165928

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0057253, 0.0054984, -0.0107976, 0.0104840
1: 0.0024777, 0.0126253, 0.0022284, 0.0126409, -0.0101632, 0.0103970
2: 0.0140326, 0.0305792, 0.0140281, 0.0310068, -0.0145896, 0.0141656
3: -0.0096271, -0.0019377, -0.0098028, -0.0019340, -0.0076931, 0.0078650
4: -0.0023095, 0.0022292, -0.0023119, 0.0023732, -0.0046827, 0.0045411
5: -0.0031822, 0.0074543, -0.0034860, 0.0074682, -0.0106504, 0.0109403
6: -0.0065304, -0.0006294, -0.0066810, -0.0005981, -0.0059322, 0.0060517
7: -0.0112415, -0.0000816, -0.0113801, 0.0001318, -0.0113733, 0.0112985
8: -0.0106450, 0.0008758, -0.0109331, 0.0008848, -0.0115298, 0.0118088
9: 0.9808148, 0.9972770, 0.9806254, 0.9976319, -0.0168171, 0.0166516

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0113225
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0113225
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057548, 0.0051352, -0.0056173, 0.0054546, -0.0108134, 0.0103675
1: 0.0025355, 0.0126286, 0.0019295, 0.0125133, -0.0099778, 0.0106991
2: 0.0140131, 0.0305146, 0.0139541, 0.0307275, -0.0142485, 0.0141319
3: -0.0096019, -0.0019275, -0.0098491, -0.0020419, -0.0075599, 0.0079216
4: -0.0022970, 0.0021768, -0.0022857, 0.0026948, -0.0049918, 0.0044625
5: -0.0031136, 0.0074671, -0.0037913, 0.0073173, -0.0104309, 0.0112585
6: -0.0064914, -0.0006393, -0.0068709, -0.0006510, -0.0058404, 0.0062316
7: -0.0111989, -0.0001393, -0.0112952, 0.0005126, -0.0117115, 0.0111558
8: -0.0106104, 0.0009088, -0.0109686, 0.0007703, -0.0113807, 0.0118774
9: 0.9808776, 0.9971525, 0.9807639, 0.9984745, -0.0175968, 0.0163886

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0068992, 0.0047563, -0.0056173, 0.0054546, -0.0119821, 0.0100526
1: 0.0027820, 0.0138329, 0.0019295, 0.0125133, -0.0097313, 0.0119035
2: 0.0126169, 0.0299911, 0.0139541, 0.0307275, -0.0156946, 0.0139723
3: -0.0093943, -0.0010534, -0.0098491, -0.0020419, -0.0073524, 0.0087957
4: -0.0026949, 0.0020032, -0.0022857, 0.0026948, -0.0053897, 0.0042888
5: -0.0027867, 0.0088744, -0.0037913, 0.0073173, -0.0101040, 0.0126657
6: -0.0063192, -0.0000224, -0.0068709, -0.0006510, -0.0056682, 0.0068485
7: -0.0117971, -0.0004077, -0.0112952, 0.0005126, -0.0123098, 0.0108875
8: -0.0102907, 0.0023268, -0.0109686, 0.0007703, -0.0110610, 0.0132954
9: 0.9799096, 0.9968352, 0.9807639, 0.9984745, -0.0185649, 0.0160713

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057548, 0.0051352, -0.0067659, 0.0050782, -0.0104974, 0.0115480
1: 0.0025355, 0.0126286, 0.0021609, 0.0137127, -0.0111773, 0.0104676
2: 0.0140131, 0.0305146, 0.0125542, 0.0302091, -0.0140534, 0.0155897
3: -0.0096019, -0.0019275, -0.0096418, -0.0011639, -0.0084380, 0.0077143
4: -0.0022970, 0.0021768, -0.0026821, 0.0025270, -0.0048241, 0.0048588
5: -0.0031136, 0.0074671, -0.0034893, 0.0087315, -0.0118451, 0.0109565
6: -0.0064914, -0.0006393, -0.0067080, -0.0000424, -0.0064489, 0.0060688
7: -0.0111989, -0.0001393, -0.0118602, 0.0002460, -0.0114449, 0.0117209
8: -0.0106104, 0.0009088, -0.0106494, 0.0021933, -0.0128037, 0.0115583
9: 0.9808776, 0.9971525, 0.9798483, 0.9981731, -0.0172954, 0.0173042

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0068992, 0.0047563, -0.0067659, 0.0050782, -0.0115486, 0.0111167
1: 0.0027820, 0.0138329, 0.0021609, 0.0137127, -0.0109308, 0.0116720
2: 0.0126169, 0.0299911, 0.0125542, 0.0302091, -0.0148455, 0.0147654
3: -0.0093943, -0.0010534, -0.0096418, -0.0011639, -0.0082305, 0.0085884
4: -0.0026949, 0.0020032, -0.0026821, 0.0025270, -0.0052220, 0.0046852
5: -0.0027867, 0.0088744, -0.0034893, 0.0087315, -0.0115183, 0.0123637
6: -0.0063192, -0.0000224, -0.0067080, -0.0000424, -0.0062768, 0.0066856
7: -0.0117971, -0.0004077, -0.0118602, 0.0002460, -0.0120431, 0.0114525
8: -0.0102907, 0.0023268, -0.0106494, 0.0021933, -0.0124839, 0.0129762
9: 0.9799096, 0.9968352, 0.9798483, 0.9981731, -0.0182635, 0.0169870

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0056490, 0.0056098, -0.0109209, 0.0103968
1: 0.0024777, 0.0126253, 0.0018637, 0.0125166, -0.0100389, 0.0107616
2: 0.0140326, 0.0305792, 0.0139295, 0.0309755, -0.0144203, 0.0140729
3: -0.0096271, -0.0019377, -0.0099271, -0.0020247, -0.0076024, 0.0079893
4: -0.0023095, 0.0022292, -0.0022726, 0.0027076, -0.0050171, 0.0045018
5: -0.0031822, 0.0074543, -0.0038639, 0.0073346, -0.0105167, 0.0113182
6: -0.0065304, -0.0006294, -0.0069023, -0.0006421, -0.0058883, 0.0062730
7: -0.0112415, -0.0000816, -0.0113526, 0.0005621, -0.0118036, 0.0112710
8: -0.0106450, 0.0008758, -0.0110881, 0.0008073, -0.0114523, 0.0119639
9: 0.9808148, 0.9972770, 0.9806902, 0.9985070, -0.0176922, 0.0165868

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0068766, 0.0048013, -0.0056480, 0.0054511, -0.0119447, 0.0101145
1: 0.0027177, 0.0138473, 0.0019878, 0.0125082, -0.0097905, 0.0118595
2: 0.0126341, 0.0300554, 0.0139311, 0.0307555, -0.0156777, 0.0140902
3: -0.0094224, -0.0010629, -0.0098412, -0.0020265, -0.0073959, 0.0087782
4: -0.0027126, 0.0020633, -0.0022716, 0.0026360, -0.0053486, 0.0043348
5: -0.0028694, 0.0088796, -0.0037250, 0.0073276, -0.0101969, 0.0126046
6: -0.0063635, -0.0000061, -0.0068301, -0.0006598, -0.0057037, 0.0068241
7: -0.0118321, -0.0003420, -0.0112750, 0.0004466, -0.0122787, 0.0109329
8: -0.0103321, 0.0023020, -0.0109526, 0.0008028, -0.0111349, 0.0132546
9: 0.9798547, 0.9969749, 0.9807976, 0.9983351, -0.0184804, 0.0161773

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0056206, 0.0056552, -0.0109974, 0.0104140
1: 0.0024777, 0.0126253, 0.0017986, 0.0125274, -0.0100497, 0.0108268
2: 0.0140326, 0.0305792, 0.0139496, 0.0310403, -0.0146253, 0.0142270
3: -0.0096271, -0.0019377, -0.0099538, -0.0020377, -0.0075894, 0.0080161
4: -0.0023095, 0.0022292, -0.0022887, 0.0027693, -0.0050788, 0.0045179
5: -0.0031822, 0.0074543, -0.0039464, 0.0073299, -0.0105121, 0.0114008
6: -0.0065304, -0.0006294, -0.0069459, -0.0006251, -0.0059052, 0.0063165
7: -0.0112415, -0.0000816, -0.0114020, 0.0006266, -0.0118681, 0.0113204
8: -0.0106450, 0.0008758, -0.0111265, 0.0007786, -0.0114235, 0.0120023
9: 0.9808148, 0.9972770, 0.9806213, 0.9986537, -0.0178388, 0.0166557

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
time: 1.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0068766, 0.0048013, -0.0056195, 0.0054984, -0.0120239, 0.0101344
1: 0.0027177, 0.0138473, 0.0019220, 0.0125188, -0.0098011, 0.0119253
2: 0.0126341, 0.0300554, 0.0139513, 0.0308209, -0.0158976, 0.0142581
3: -0.0094224, -0.0010629, -0.0098687, -0.0020396, -0.0073828, 0.0088058
4: -0.0027126, 0.0020633, -0.0022876, 0.0026979, -0.0054105, 0.0043509
5: -0.0028694, 0.0088796, -0.0038087, 0.0073226, -0.0101920, 0.0126883
6: -0.0063635, -0.0000061, -0.0068758, -0.0006432, -0.0057203, 0.0068697
7: -0.0118321, -0.0003420, -0.0113237, 0.0005116, -0.0123437, 0.0109817
8: -0.0103321, 0.0023020, -0.0109929, 0.0007739, -0.0111060, 0.0132949
9: 0.9798547, 0.9969749, 0.9807264, 0.9984813, -0.0186266, 0.0162486

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
time: 3.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0057227, 0.0053017, -0.0105387, 0.0106353
1: 0.0021074, 0.0124992, 0.0023583, 0.0126287, -0.0105213, 0.0101408
2: 0.0139343, 0.0305487, 0.0140321, 0.0306935, -0.0142707, 0.0141083
3: -0.0097542, -0.0020294, -0.0096967, -0.0019372, -0.0078169, 0.0076673
4: -0.0022694, 0.0025633, -0.0023097, 0.0023047, -0.0045741, 0.0048730
5: -0.0035617, 0.0073197, -0.0033384, 0.0074571, -0.0110188, 0.0106581
6: -0.0067548, -0.0006758, -0.0066049, -0.0006216, -0.0061332, 0.0059291
7: -0.0112094, 0.0003534, -0.0112777, 0.0000192, -0.0112287, 0.0116310
8: -0.0108022, 0.0007975, -0.0107755, 0.0008776, -0.0116798, 0.0115730
9: 0.9808887, 0.9981471, 0.9807646, 0.9974712, -0.0165825, 0.0173825

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0057227, 0.0053017, -0.0117094, 0.0103219
1: 0.0023423, 0.0136958, 0.0023583, 0.0126287, -0.0102864, 0.0113374
2: 0.0125382, 0.0300326, 0.0140321, 0.0306935, -0.0157223, 0.0139350
3: -0.0095456, -0.0011591, -0.0096967, -0.0019372, -0.0076084, 0.0085376
4: -0.0026604, 0.0023924, -0.0023097, 0.0023047, -0.0049650, 0.0047020
5: -0.0032551, 0.0087230, -0.0033384, 0.0074571, -0.0107122, 0.0120614
6: -0.0065894, -0.0000714, -0.0066049, -0.0006216, -0.0059677, 0.0065335
7: -0.0117850, 0.0000863, -0.0112777, 0.0000192, -0.0118042, 0.0113640
8: -0.0104798, 0.0022140, -0.0107755, 0.0008776, -0.0113574, 0.0129895
9: 0.9799623, 0.9978392, 0.9807646, 0.9974712, -0.0175088, 0.0170746

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0068767, 0.0049221, -0.0102217, 0.0118112
1: 0.0021074, 0.0124992, 0.0026041, 0.0138501, -0.0117427, 0.0098951
2: 0.0139343, 0.0305487, 0.0126336, 0.0301669, -0.0140891, 0.0155601
3: -0.0097542, -0.0020294, -0.0094892, -0.0010624, -0.0086917, 0.0074598
4: -0.0022694, 0.0025633, -0.0027127, 0.0021342, -0.0044037, 0.0052760
5: -0.0035617, 0.0073197, -0.0030180, 0.0088819, -0.0124436, 0.0103377
6: -0.0067548, -0.0006758, -0.0064352, -0.0000001, -0.0067547, 0.0057594
7: -0.0112094, 0.0003534, -0.0118616, -0.0002472, -0.0109622, 0.0122149
8: -0.0108022, 0.0007975, -0.0104579, 0.0023033, -0.0131055, 0.0112555
9: 0.9808887, 0.9981471, 0.9798118, 0.9971586, -0.0162699, 0.0183353

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125168
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125168
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0068767, 0.0049221, -0.0112743, 0.0113799
1: 0.0023423, 0.0136958, 0.0026041, 0.0138501, -0.0115078, 0.0110917
2: 0.0125382, 0.0300326, 0.0126336, 0.0301669, -0.0148874, 0.0147195
3: -0.0095456, -0.0011591, -0.0094892, -0.0010624, -0.0084832, 0.0083301
4: -0.0026604, 0.0023924, -0.0027127, 0.0021342, -0.0047946, 0.0051051
5: -0.0032551, 0.0087230, -0.0030180, 0.0088819, -0.0121370, 0.0117410
6: -0.0065894, -0.0000714, -0.0064352, -0.0000001, -0.0065893, 0.0063639
7: -0.0117850, 0.0000863, -0.0118616, -0.0002472, -0.0115377, 0.0119479
8: -0.0104798, 0.0022140, -0.0104579, 0.0023033, -0.0127832, 0.0126719
9: 0.9799623, 0.9978392, 0.9798118, 0.9971586, -0.0171962, 0.0180274

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125193
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125193
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0057575, 0.0054564, -0.0106528, 0.0106717
1: 0.0020438, 0.0125097, 0.0022875, 0.0126435, -0.0105997, 0.0102222
2: 0.0139545, 0.0306127, 0.0140086, 0.0309421, -0.0144445, 0.0140487
3: -0.0097794, -0.0020425, -0.0097759, -0.0019237, -0.0078558, 0.0077334
4: -0.0022855, 0.0026249, -0.0022995, 0.0023202, -0.0046057, 0.0049244
5: -0.0036424, 0.0073144, -0.0034181, 0.0074809, -0.0111234, 0.0107325
6: -0.0067977, -0.0006594, -0.0066403, -0.0006086, -0.0061892, 0.0059809
7: -0.0112578, 0.0004158, -0.0113376, 0.0000733, -0.0113311, 0.0117534
8: -0.0108384, 0.0007684, -0.0108967, 0.0009173, -0.0117557, 0.0116652
9: 0.9808157, 0.9982911, 0.9806842, 0.9975065, -0.0166908, 0.0176070

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126667
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126667
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0057566, 0.0052978, -0.0116793, 0.0103928
1: 0.0022689, 0.0137097, 0.0024126, 0.0126361, -0.0103672, 0.0112971
2: 0.0125547, 0.0300980, 0.0140101, 0.0307215, -0.0157099, 0.0140480
3: -0.0095757, -0.0011644, -0.0096892, -0.0019252, -0.0076505, 0.0085247
4: -0.0026819, 0.0024590, -0.0022987, 0.0022469, -0.0049288, 0.0047577
5: -0.0033472, 0.0087291, -0.0032760, 0.0074742, -0.0108214, 0.0120051
6: -0.0066386, -0.0000491, -0.0065691, -0.0006251, -0.0060135, 0.0065200
7: -0.0118299, 0.0001518, -0.0112614, -0.0000402, -0.0117897, 0.0114132
8: -0.0105247, 0.0021918, -0.0107622, 0.0009133, -0.0114380, 0.0129539
9: 0.9798938, 0.9979986, 0.9807898, 0.9973302, -0.0174364, 0.0172088

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
time: 1.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
time: 2.29 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0057253, 0.0054984, -0.0107268, 0.0106839
1: 0.0020438, 0.0125097, 0.0022284, 0.0126409, -0.0105970, 0.0102814
2: 0.0139545, 0.0306127, 0.0140281, 0.0310068, -0.0146505, 0.0142008
3: -0.0097794, -0.0020425, -0.0098028, -0.0019340, -0.0078455, 0.0077603
4: -0.0022855, 0.0026249, -0.0023119, 0.0023732, -0.0046587, 0.0049368
5: -0.0036424, 0.0073144, -0.0034860, 0.0074682, -0.0111107, 0.0108003
6: -0.0067977, -0.0006594, -0.0066810, -0.0005981, -0.0061996, 0.0060216
7: -0.0112578, 0.0004158, -0.0113801, 0.0001318, -0.0113896, 0.0117959
8: -0.0108384, 0.0007684, -0.0109331, 0.0008848, -0.0117232, 0.0117015
9: 0.9808157, 0.9982911, 0.9806254, 0.9976319, -0.0168162, 0.0176657

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126667
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126667
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0057245, 0.0053406, -0.0117573, 0.0104075
1: 0.0022689, 0.0137097, 0.0023526, 0.0126332, -0.0103642, 0.0113571
2: 0.0125547, 0.0300980, 0.0140296, 0.0307867, -0.0159281, 0.0142159
3: -0.0095757, -0.0011644, -0.0097171, -0.0019354, -0.0076403, 0.0085526
4: -0.0026819, 0.0024590, -0.0023111, 0.0023007, -0.0049826, 0.0047701
5: -0.0033472, 0.0087291, -0.0033472, 0.0074614, -0.0108086, 0.0120763
6: -0.0066386, -0.0000491, -0.0066098, -0.0006149, -0.0060237, 0.0065607
7: -0.0118299, 0.0001518, -0.0113042, 0.0000200, -0.0118499, 0.0114560
8: -0.0105247, 0.0021918, -0.0107994, 0.0008805, -0.0114052, 0.0129912
9: 0.9798938, 0.9979986, 0.9807292, 0.9974561, -0.0175623, 0.0172694

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126742
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126742
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0056173, 0.0054546, -0.0107811, 0.0106101
1: 0.0021074, 0.0124992, 0.0019295, 0.0125133, -0.0104059, 0.0105697
2: 0.0139343, 0.0305487, 0.0139541, 0.0307275, -0.0145192, 0.0143892
3: -0.0097542, -0.0020294, -0.0098491, -0.0020419, -0.0077122, 0.0078197
4: -0.0022694, 0.0025633, -0.0022857, 0.0026948, -0.0049642, 0.0048490
5: -0.0035617, 0.0073197, -0.0037913, 0.0073173, -0.0108790, 0.0111110
6: -0.0067548, -0.0006758, -0.0068709, -0.0006510, -0.0061038, 0.0061951
7: -0.0112094, 0.0003534, -0.0112952, 0.0005126, -0.0117221, 0.0116485
8: -0.0108022, 0.0007975, -0.0109686, 0.0007703, -0.0115725, 0.0117661
9: 0.9808887, 0.9981471, 0.9807639, 0.9984745, -0.0175858, 0.0173832

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0056173, 0.0054546, -0.0119496, 0.0102986
1: 0.0023423, 0.0136958, 0.0019295, 0.0125133, -0.0101710, 0.0117663
2: 0.0125382, 0.0300326, 0.0139541, 0.0307275, -0.0159531, 0.0141929
3: -0.0095456, -0.0011591, -0.0098491, -0.0020419, -0.0075037, 0.0086900
4: -0.0026604, 0.0023924, -0.0022857, 0.0026948, -0.0053551, 0.0046781
5: -0.0032551, 0.0087230, -0.0037913, 0.0073173, -0.0105724, 0.0125143
6: -0.0065894, -0.0000714, -0.0068709, -0.0006510, -0.0059384, 0.0067995
7: -0.0117850, 0.0000863, -0.0112952, 0.0005126, -0.0122976, 0.0113814
8: -0.0104798, 0.0022140, -0.0109686, 0.0007703, -0.0112502, 0.0131826
9: 0.9799623, 0.9978392, 0.9807639, 0.9984745, -0.0185121, 0.0170753

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0067659, 0.0050782, -0.0104675, 0.0117892
1: 0.0021074, 0.0124992, 0.0021609, 0.0137127, -0.0116054, 0.0103382
2: 0.0139343, 0.0305487, 0.0125542, 0.0302091, -0.0143079, 0.0158270
3: -0.0097542, -0.0020294, -0.0096418, -0.0011639, -0.0085903, 0.0076124
4: -0.0022694, 0.0025633, -0.0026821, 0.0025270, -0.0047965, 0.0052454
5: -0.0035617, 0.0073197, -0.0034893, 0.0087315, -0.0122933, 0.0108090
6: -0.0067548, -0.0006758, -0.0067080, -0.0000424, -0.0067124, 0.0060323
7: -0.0112094, 0.0003534, -0.0118602, 0.0002460, -0.0114554, 0.0122136
8: -0.0108022, 0.0007975, -0.0106494, 0.0021933, -0.0129955, 0.0114470
9: 0.9808887, 0.9981471, 0.9798483, 0.9981731, -0.0172844, 0.0182989

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126129
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126129
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0067659, 0.0050782, -0.0115186, 0.0113607
1: 0.0023423, 0.0136958, 0.0021609, 0.0137127, -0.0113705, 0.0115348
2: 0.0125382, 0.0300326, 0.0125542, 0.0302091, -0.0151154, 0.0149954
3: -0.0095456, -0.0011591, -0.0096418, -0.0011639, -0.0083817, 0.0084827
4: -0.0026604, 0.0023924, -0.0026821, 0.0025270, -0.0051874, 0.0050744
5: -0.0032551, 0.0087230, -0.0034893, 0.0087315, -0.0119867, 0.0122123
6: -0.0065894, -0.0000714, -0.0067080, -0.0000424, -0.0065469, 0.0066367
7: -0.0117850, 0.0000863, -0.0118602, 0.0002460, -0.0120310, 0.0119465
8: -0.0104798, 0.0022140, -0.0106494, 0.0021933, -0.0126731, 0.0128634
9: 0.9799623, 0.9978392, 0.9798483, 0.9981731, -0.0182107, 0.0179909

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126266
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126266
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0056508, 0.0056098, -0.0108946, 0.0106821
1: 0.0020438, 0.0125097, 0.0018636, 0.0125181, -0.0104743, 0.0106461
2: 0.0139545, 0.0306127, 0.0139272, 0.0309755, -0.0146970, 0.0145176
3: -0.0097794, -0.0020425, -0.0099271, -0.0020233, -0.0077561, 0.0078846
4: -0.0022855, 0.0026249, -0.0022731, 0.0027077, -0.0049932, 0.0048980
5: -0.0036424, 0.0073144, -0.0038639, 0.0073364, -0.0109789, 0.0111783
6: -0.0067977, -0.0006594, -0.0069024, -0.0006412, -0.0061565, 0.0062430
7: -0.0112578, 0.0004158, -0.0113537, 0.0005622, -0.0118200, 0.0117695
8: -0.0108384, 0.0007684, -0.0110881, 0.0008096, -0.0116480, 0.0118566
9: 0.9808157, 0.9982911, 0.9806883, 0.9985073, -0.0176916, 0.0176028

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127538
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127538
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0056480, 0.0054511, -0.0119216, 0.0103635
1: 0.0022689, 0.0137097, 0.0019878, 0.0125082, -0.0102392, 0.0117219
2: 0.0125547, 0.0300980, 0.0139311, 0.0307555, -0.0159392, 0.0143317
3: -0.0095757, -0.0011644, -0.0098412, -0.0020265, -0.0075492, 0.0086767
4: -0.0026819, 0.0024590, -0.0022716, 0.0026360, -0.0053179, 0.0047306
5: -0.0033472, 0.0087291, -0.0037250, 0.0073276, -0.0106748, 0.0124541
6: -0.0066386, -0.0000491, -0.0068301, -0.0006598, -0.0059787, 0.0067810
7: -0.0118299, 0.0001518, -0.0112750, 0.0004466, -0.0122765, 0.0114268
8: -0.0105247, 0.0021918, -0.0109526, 0.0008028, -0.0113275, 0.0131443
9: 0.9798938, 0.9979986, 0.9807976, 0.9983351, -0.0184413, 0.0172009

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127808
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127808
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0056210, 0.0056552, -0.0109739, 0.0106957
1: 0.0020438, 0.0125097, 0.0017986, 0.0125278, -0.0104840, 0.0107111
2: 0.0139545, 0.0306127, 0.0139490, 0.0310403, -0.0149144, 0.0146484
3: -0.0097794, -0.0020425, -0.0099538, -0.0020374, -0.0077421, 0.0079113
4: -0.0022855, 0.0026249, -0.0022889, 0.0027694, -0.0050548, 0.0049137
5: -0.0036424, 0.0073144, -0.0039464, 0.0073304, -0.0109728, 0.0112608
6: -0.0067977, -0.0006594, -0.0069459, -0.0006249, -0.0061728, 0.0062865
7: -0.0112578, 0.0004158, -0.0114023, 0.0006266, -0.0118844, 0.0118181
8: -0.0108384, 0.0007684, -0.0111265, 0.0007791, -0.0116176, 0.0118949
9: 0.9808157, 0.9982911, 0.9806208, 0.9986536, -0.0178379, 0.0176703

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127538
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127538
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0056195, 0.0054984, -0.0120035, 0.0103869
1: 0.0022689, 0.0137097, 0.0019220, 0.0125188, -0.0102499, 0.0117877
2: 0.0125547, 0.0300980, 0.0139513, 0.0308209, -0.0161651, 0.0145052
3: -0.0095757, -0.0011644, -0.0098687, -0.0020396, -0.0075361, 0.0087043
4: -0.0026819, 0.0024590, -0.0022876, 0.0026979, -0.0053798, 0.0047466
5: -0.0033472, 0.0087291, -0.0038087, 0.0073226, -0.0106698, 0.0125378
6: -0.0066386, -0.0000491, -0.0068758, -0.0006432, -0.0059954, 0.0068267
7: -0.0118299, 0.0001518, -0.0113237, 0.0005116, -0.0123415, 0.0114755
8: -0.0105247, 0.0021918, -0.0109929, 0.0007739, -0.0112986, 0.0131847
9: 0.9798938, 0.9979986, 0.9807264, 0.9984813, -0.0185875, 0.0172722

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127808
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127808
time: 0.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.72 seconds
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
NS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0113225
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0113225
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0113225
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125879, upper bound: 0.0116882
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125529, upper bound: 0.0112916
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125168, upper bound: 0.0113225
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0125197, upper bound: 0.0113225
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0113932, upper bound: 0.0128118
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125168
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125168
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125193
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112921, upper bound: 0.0125193
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126667
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126667
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112916, upper bound: 0.0126742
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126667
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126667
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126742
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0112923, upper bound: 0.0126742
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118322, upper bound: 0.0128455
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126129
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126129
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126266
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118248, upper bound: 0.0126266
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127538
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127538
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127808
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118245, upper bound: 0.0127808
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127538
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127538
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127808
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 9, lower bound: -0.0118295, upper bound: 0.0127808

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057548, 0.0051352, -0.0056457, 0.0052875, -0.0106155, 0.0103418
1: 0.0025355, 0.0126286, 0.0021074, 0.0124992, -0.0099637, 0.0105212
2: 0.0140131, 0.0305146, 0.0139343, 0.0305487, -0.0139663, 0.0139899
3: -0.0096019, -0.0019275, -0.0097542, -0.0020294, -0.0075725, 0.0078266
4: -0.0022970, 0.0021768, -0.0022694, 0.0025633, -0.0048603, 0.0044462
5: -0.0031136, 0.0074671, -0.0035617, 0.0073197, -0.0104333, 0.0110289
6: -0.0064914, -0.0006393, -0.0067548, -0.0006758, -0.0058156, 0.0061155
7: -0.0111989, -0.0001393, -0.0112094, 0.0003534, -0.0115523, 0.0110701
8: -0.0106104, 0.0009088, -0.0108022, 0.0007975, -0.0114079, 0.0117110
9: 0.9808776, 0.9971525, 0.9808887, 0.9981471, -0.0172695, 0.0162638

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128861, upper bound: 0.0115223
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115161
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057548, 0.0051352, -0.0056172, 0.0053314, -0.0106678, 0.0103375
1: 0.0025355, 0.0126286, 0.0020438, 0.0125097, -0.0099742, 0.0105847
2: 0.0140131, 0.0305146, 0.0139545, 0.0306127, -0.0140369, 0.0140090
3: -0.0096019, -0.0019275, -0.0097794, -0.0020425, -0.0075593, 0.0078519
4: -0.0022970, 0.0021768, -0.0022855, 0.0026249, -0.0049219, 0.0044623
5: -0.0031136, 0.0074671, -0.0036424, 0.0073144, -0.0104280, 0.0111096
6: -0.0064914, -0.0006393, -0.0067977, -0.0006594, -0.0058319, 0.0061585
7: -0.0111989, -0.0001393, -0.0112578, 0.0004158, -0.0116147, 0.0111185
8: -0.0106104, 0.0009088, -0.0108384, 0.0007684, -0.0113788, 0.0117473
9: 0.9808776, 0.9971525, 0.9808157, 0.9982911, -0.0174135, 0.0163368

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128861, upper bound: 0.0115223
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115161
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0068992, 0.0047563, -0.0056457, 0.0052875, -0.0117842, 0.0100268
1: 0.0027820, 0.0138329, 0.0021074, 0.0124992, -0.0097172, 0.0117255
2: 0.0126169, 0.0299911, 0.0139343, 0.0305487, -0.0154125, 0.0138303
3: -0.0093943, -0.0010534, -0.0097542, -0.0020294, -0.0073650, 0.0087007
4: -0.0026949, 0.0020032, -0.0022694, 0.0025633, -0.0052583, 0.0042726
5: -0.0027867, 0.0088744, -0.0035617, 0.0073197, -0.0101064, 0.0124361
6: -0.0063192, -0.0000224, -0.0067548, -0.0006758, -0.0056434, 0.0067324
7: -0.0117971, -0.0004077, -0.0112094, 0.0003534, -0.0121505, 0.0108017
8: -0.0102907, 0.0023268, -0.0108022, 0.0007975, -0.0110882, 0.0131290
9: 0.9799096, 0.9968352, 0.9808887, 0.9981471, -0.0182375, 0.0159466

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121593, upper bound: 0.0107825
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114485, upper bound: 0.0106195
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068992, 0.0047563, -0.0056172, 0.0053314, -0.0118364, 0.0100225
1: 0.0027820, 0.0138329, 0.0020438, 0.0125097, -0.0097277, 0.0117891
2: 0.0126169, 0.0299911, 0.0139545, 0.0306127, -0.0154831, 0.0138494
3: -0.0093943, -0.0010534, -0.0097794, -0.0020425, -0.0073518, 0.0087260
4: -0.0026949, 0.0020032, -0.0022855, 0.0026249, -0.0053198, 0.0042887
5: -0.0027867, 0.0088744, -0.0036424, 0.0073144, -0.0101011, 0.0125168
6: -0.0063192, -0.0000224, -0.0067977, -0.0006594, -0.0056598, 0.0067753
7: -0.0117971, -0.0004077, -0.0112578, 0.0004158, -0.0122129, 0.0108501
8: -0.0102907, 0.0023268, -0.0108384, 0.0007684, -0.0110591, 0.0131652
9: 0.9799096, 0.9968352, 0.9808157, 0.9982911, -0.0183815, 0.0160195

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121593, upper bound: 0.0107825
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114485, upper bound: 0.0106195
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057548, 0.0051352, -0.0067875, 0.0049107, -0.0103021, 0.0115125
1: 0.0025355, 0.0126286, 0.0023423, 0.0136958, -0.0111603, 0.0102863
2: 0.0140131, 0.0305146, 0.0125382, 0.0300326, -0.0137930, 0.0154415
3: -0.0096019, -0.0019275, -0.0095456, -0.0011591, -0.0084428, 0.0076181
4: -0.0022970, 0.0021768, -0.0026604, 0.0023924, -0.0046894, 0.0048372
5: -0.0031136, 0.0074671, -0.0032551, 0.0087230, -0.0118366, 0.0107223
6: -0.0064914, -0.0006393, -0.0065894, -0.0000714, -0.0064200, 0.0059501
7: -0.0111989, -0.0001393, -0.0117850, 0.0000863, -0.0112852, 0.0116456
8: -0.0106104, 0.0009088, -0.0104798, 0.0022140, -0.0128244, 0.0113887
9: 0.9808776, 0.9971525, 0.9799623, 0.9978392, -0.0169616, 0.0171902

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124687, upper bound: 0.0102669
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057548, 0.0051352, -0.0067659, 0.0049596, -0.0103587, 0.0115167
1: 0.0025355, 0.0126286, 0.0022689, 0.0137097, -0.0111742, 0.0103596
2: 0.0140131, 0.0305146, 0.0125547, 0.0300980, -0.0138623, 0.0154632
3: -0.0096019, -0.0019275, -0.0095757, -0.0011644, -0.0084374, 0.0076482
4: -0.0022970, 0.0021768, -0.0026819, 0.0024590, -0.0047560, 0.0048587
5: -0.0031136, 0.0074671, -0.0033472, 0.0087291, -0.0118427, 0.0108143
6: -0.0064914, -0.0006393, -0.0066386, -0.0000491, -0.0064423, 0.0059993
7: -0.0111989, -0.0001393, -0.0118299, 0.0001518, -0.0113507, 0.0116906
8: -0.0106104, 0.0009088, -0.0105247, 0.0021918, -0.0128021, 0.0114335
9: 0.9808776, 0.9971525, 0.9798938, 0.9979986, -0.0171210, 0.0172587

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124687, upper bound: 0.0102669
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0068992, 0.0047563, -0.0067875, 0.0049107, -0.0113525, 0.0110793
1: 0.0027820, 0.0138329, 0.0023423, 0.0136958, -0.0109138, 0.0114906
2: 0.0126169, 0.0299911, 0.0125382, 0.0300326, -0.0145798, 0.0146251
3: -0.0093943, -0.0010534, -0.0095456, -0.0011591, -0.0082352, 0.0084922
4: -0.0026949, 0.0020032, -0.0026604, 0.0023924, -0.0050873, 0.0046636
5: -0.0027867, 0.0088744, -0.0032551, 0.0087230, -0.0115097, 0.0121295
6: -0.0063192, -0.0000224, -0.0065894, -0.0000714, -0.0062479, 0.0065670
7: -0.0117971, -0.0004077, -0.0117850, 0.0000863, -0.0118834, 0.0113773
8: -0.0102907, 0.0023268, -0.0104798, 0.0022140, -0.0125047, 0.0128067
9: 0.9799096, 0.9968352, 0.9799623, 0.9978392, -0.0179296, 0.0168729

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120518, upper bound: 0.0100636
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112642, upper bound: 0.0099079
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068992, 0.0047563, -0.0067659, 0.0049596, -0.0114105, 0.0110853
1: 0.0027820, 0.0138329, 0.0022689, 0.0137097, -0.0109277, 0.0115640
2: 0.0126169, 0.0299911, 0.0125547, 0.0300980, -0.0146501, 0.0146430
3: -0.0093943, -0.0010534, -0.0095757, -0.0011644, -0.0082299, 0.0085223
4: -0.0026949, 0.0020032, -0.0026819, 0.0024590, -0.0051539, 0.0046851
5: -0.0027867, 0.0088744, -0.0033472, 0.0087291, -0.0115158, 0.0122216
6: -0.0063192, -0.0000224, -0.0066386, -0.0000491, -0.0062701, 0.0066162
7: -0.0117971, -0.0004077, -0.0118299, 0.0001518, -0.0119489, 0.0114222
8: -0.0102907, 0.0023268, -0.0105247, 0.0021918, -0.0124824, 0.0128515
9: 0.9799096, 0.9968352, 0.9798938, 0.9979986, -0.0180890, 0.0169414

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120518, upper bound: 0.0100636
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112642, upper bound: 0.0099079
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0056457, 0.0052875, -0.0106057, 0.0103918
1: 0.0024777, 0.0126253, 0.0021074, 0.0124992, -0.0100215, 0.0105180
2: 0.0140326, 0.0305792, 0.0139343, 0.0305487, -0.0139853, 0.0140603
3: -0.0096271, -0.0019377, -0.0097542, -0.0020294, -0.0075977, 0.0078164
4: -0.0023095, 0.0022292, -0.0022694, 0.0025633, -0.0048728, 0.0044986
5: -0.0031822, 0.0074543, -0.0035617, 0.0073197, -0.0105018, 0.0110161
6: -0.0065304, -0.0006294, -0.0067548, -0.0006758, -0.0058546, 0.0061254
7: -0.0112415, -0.0000816, -0.0112094, 0.0003534, -0.0115949, 0.0111278
8: -0.0106450, 0.0008758, -0.0108022, 0.0007975, -0.0114425, 0.0116780
9: 0.9808148, 0.9972770, 0.9808887, 0.9981471, -0.0173323, 0.0163883

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124005, upper bound: 0.0102878
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0067875, 0.0049107, -0.0102922, 0.0115625
1: 0.0024777, 0.0126253, 0.0023423, 0.0136958, -0.0112181, 0.0102831
2: 0.0140326, 0.0305792, 0.0125382, 0.0300326, -0.0138120, 0.0155119
3: -0.0096271, -0.0019377, -0.0095456, -0.0011591, -0.0084680, 0.0076079
4: -0.0023095, 0.0022292, -0.0026604, 0.0023924, -0.0047019, 0.0048896
5: -0.0031822, 0.0074543, -0.0032551, 0.0087230, -0.0119052, 0.0107095
6: -0.0065304, -0.0006294, -0.0065894, -0.0000714, -0.0064590, 0.0059600
7: -0.0112415, -0.0000816, -0.0117850, 0.0000863, -0.0113278, 0.0117034
8: -0.0106450, 0.0008758, -0.0104798, 0.0022140, -0.0128590, 0.0113556
9: 0.9808148, 0.9972770, 0.9799623, 0.9978392, -0.0170244, 0.0173146

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124005, upper bound: 0.0102878
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0068766, 0.0048013, -0.0056457, 0.0052875, -0.0117808, 0.0100811
1: 0.0027177, 0.0138473, 0.0021074, 0.0124992, -0.0097815, 0.0117399
2: 0.0126341, 0.0300554, 0.0139343, 0.0305487, -0.0154332, 0.0138978
3: -0.0094224, -0.0010629, -0.0097542, -0.0020294, -0.0073930, 0.0086912
4: -0.0027126, 0.0020633, -0.0022694, 0.0025633, -0.0052759, 0.0043327
5: -0.0028694, 0.0088796, -0.0035617, 0.0073197, -0.0101890, 0.0124413
6: -0.0063635, -0.0000061, -0.0067548, -0.0006758, -0.0056877, 0.0067487
7: -0.0118321, -0.0003420, -0.0112094, 0.0003534, -0.0121855, 0.0108674
8: -0.0103321, 0.0023020, -0.0108022, 0.0007975, -0.0111296, 0.0131042
9: 0.9798547, 0.9969749, 0.9808887, 0.9981471, -0.0182924, 0.0160863

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120126, upper bound: 0.0100939
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112389, upper bound: 0.0099244
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068766, 0.0048013, -0.0067875, 0.0049107, -0.0113495, 0.0111333
1: 0.0027177, 0.0138473, 0.0023423, 0.0136958, -0.0109781, 0.0115050
2: 0.0126341, 0.0300554, 0.0125382, 0.0300326, -0.0145975, 0.0146936
3: -0.0094224, -0.0010629, -0.0095456, -0.0011591, -0.0082633, 0.0084827
4: -0.0027126, 0.0020633, -0.0026604, 0.0023924, -0.0051050, 0.0047237
5: -0.0028694, 0.0088796, -0.0032551, 0.0087230, -0.0115924, 0.0121347
6: -0.0063635, -0.0000061, -0.0065894, -0.0000714, -0.0062922, 0.0065833
7: -0.0118321, -0.0003420, -0.0117850, 0.0000863, -0.0119184, 0.0114429
8: -0.0103321, 0.0023020, -0.0104798, 0.0022140, -0.0125461, 0.0127819
9: 0.9798547, 0.9969749, 0.9799623, 0.9978392, -0.0179845, 0.0170126

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120126, upper bound: 0.0100939
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112389, upper bound: 0.0099244
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0056172, 0.0053314, -0.0106800, 0.0104095
1: 0.0024777, 0.0126253, 0.0020438, 0.0125097, -0.0100320, 0.0105815
2: 0.0140326, 0.0305792, 0.0139545, 0.0306127, -0.0141892, 0.0142149
3: -0.0096271, -0.0019377, -0.0097794, -0.0020425, -0.0075846, 0.0078417
4: -0.0023095, 0.0022292, -0.0022855, 0.0026249, -0.0049344, 0.0045147
5: -0.0031822, 0.0074543, -0.0036424, 0.0073144, -0.0104965, 0.0110968
6: -0.0065304, -0.0006294, -0.0067977, -0.0006594, -0.0058710, 0.0061684
7: -0.0112415, -0.0000816, -0.0112578, 0.0004158, -0.0116573, 0.0111762
8: -0.0106450, 0.0008758, -0.0108384, 0.0007684, -0.0114134, 0.0117142
9: 0.9808148, 0.9972770, 0.9808157, 0.9982911, -0.0174763, 0.0164613

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124010, upper bound: 0.0102878
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057226, 0.0051777, -0.0067659, 0.0049596, -0.0103741, 0.0115905
1: 0.0024777, 0.0126253, 0.0022689, 0.0137097, -0.0112320, 0.0103564
2: 0.0140326, 0.0305792, 0.0125547, 0.0300980, -0.0140309, 0.0156782
3: -0.0096271, -0.0019377, -0.0095757, -0.0011644, -0.0084627, 0.0076379
4: -0.0023095, 0.0022292, -0.0026819, 0.0024590, -0.0047685, 0.0049111
5: -0.0031822, 0.0074543, -0.0033472, 0.0087291, -0.0119113, 0.0108015
6: -0.0065304, -0.0006294, -0.0066386, -0.0000491, -0.0064813, 0.0060092
7: -0.0112415, -0.0000816, -0.0118299, 0.0001518, -0.0113933, 0.0117483
8: -0.0106450, 0.0008758, -0.0105247, 0.0021918, -0.0128367, 0.0114005
9: 0.9808148, 0.9972770, 0.9798938, 0.9979986, -0.0171838, 0.0173832

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124010, upper bound: 0.0102878
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0068766, 0.0048013, -0.0056172, 0.0053314, -0.0118568, 0.0101014
1: 0.0027177, 0.0138473, 0.0020438, 0.0125097, -0.0097920, 0.0118035
2: 0.0126341, 0.0300554, 0.0139545, 0.0306127, -0.0156469, 0.0140668
3: -0.0094224, -0.0010629, -0.0097794, -0.0020425, -0.0073799, 0.0087165
4: -0.0027126, 0.0020633, -0.0022855, 0.0026249, -0.0053375, 0.0043488
5: -0.0028694, 0.0088796, -0.0036424, 0.0073144, -0.0101837, 0.0125220
6: -0.0063635, -0.0000061, -0.0067977, -0.0006594, -0.0057041, 0.0067917
7: -0.0118321, -0.0003420, -0.0112578, 0.0004158, -0.0122479, 0.0109158
8: -0.0103321, 0.0023020, -0.0108384, 0.0007684, -0.0111005, 0.0131405
9: 0.9798547, 0.9969749, 0.9808157, 0.9982911, -0.0184364, 0.0161592

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120153, upper bound: 0.0100939
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112417, upper bound: 0.0099244
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068766, 0.0048013, -0.0067659, 0.0049596, -0.0114337, 0.0111657
1: 0.0027177, 0.0138473, 0.0022689, 0.0137097, -0.0109920, 0.0115784
2: 0.0126341, 0.0300554, 0.0125547, 0.0300980, -0.0148159, 0.0148606
3: -0.0094224, -0.0010629, -0.0095757, -0.0011644, -0.0082580, 0.0085128
4: -0.0027126, 0.0020633, -0.0026819, 0.0024590, -0.0051716, 0.0047452
5: -0.0028694, 0.0088796, -0.0033472, 0.0087291, -0.0115985, 0.0122268
6: -0.0063635, -0.0000061, -0.0066386, -0.0000491, -0.0063144, 0.0066325
7: -0.0118321, -0.0003420, -0.0118299, 0.0001518, -0.0119839, 0.0114879
8: -0.0103321, 0.0023020, -0.0105247, 0.0021918, -0.0125239, 0.0128267
9: 0.9798547, 0.9969749, 0.9798938, 0.9979986, -0.0181439, 0.0170811

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120153, upper bound: 0.0100939
time: 2.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112417, upper bound: 0.0099244
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0057548, 0.0051352, -0.0103418, 0.0106155
1: 0.0021074, 0.0124992, 0.0025355, 0.0126286, -0.0105212, 0.0099637
2: 0.0139343, 0.0305487, 0.0140131, 0.0305146, -0.0139899, 0.0139663
3: -0.0097542, -0.0020294, -0.0096019, -0.0019275, -0.0078266, 0.0075725
4: -0.0022694, 0.0025633, -0.0022970, 0.0021768, -0.0044462, 0.0048603
5: -0.0035617, 0.0073197, -0.0031136, 0.0074671, -0.0110289, 0.0104333
6: -0.0067548, -0.0006758, -0.0064914, -0.0006393, -0.0061155, 0.0058156
7: -0.0112094, 0.0003534, -0.0111989, -0.0001393, -0.0110701, 0.0115523
8: -0.0108022, 0.0007975, -0.0106104, 0.0009088, -0.0117110, 0.0114079
9: 0.9808887, 0.9981471, 0.9808776, 0.9971525, -0.0162638, 0.0172695

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118022, upper bound: 0.0126770
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0126770
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0057226, 0.0051777, -0.0103918, 0.0106057
1: 0.0021074, 0.0124992, 0.0024777, 0.0126253, -0.0105180, 0.0100215
2: 0.0139343, 0.0305487, 0.0140326, 0.0305792, -0.0140603, 0.0139853
3: -0.0097542, -0.0020294, -0.0096271, -0.0019377, -0.0078164, 0.0075977
4: -0.0022694, 0.0025633, -0.0023095, 0.0022292, -0.0044986, 0.0048728
5: -0.0035617, 0.0073197, -0.0031822, 0.0074543, -0.0110161, 0.0105018
6: -0.0067548, -0.0006758, -0.0065304, -0.0006294, -0.0061254, 0.0058546
7: -0.0112094, 0.0003534, -0.0112415, -0.0000816, -0.0111278, 0.0115949
8: -0.0108022, 0.0007975, -0.0106450, 0.0008758, -0.0116780, 0.0114425
9: 0.9808887, 0.9981471, 0.9808148, 0.9972770, -0.0163883, 0.0173323

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118022, upper bound: 0.0126770
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0126770
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0057548, 0.0051352, -0.0115125, 0.0103021
1: 0.0023423, 0.0136958, 0.0025355, 0.0126286, -0.0102863, 0.0111603
2: 0.0125382, 0.0300326, 0.0140131, 0.0305146, -0.0154415, 0.0137930
3: -0.0095456, -0.0011591, -0.0096019, -0.0019275, -0.0076181, 0.0084428
4: -0.0026604, 0.0023924, -0.0022970, 0.0021768, -0.0048372, 0.0046894
5: -0.0032551, 0.0087230, -0.0031136, 0.0074671, -0.0107223, 0.0118366
6: -0.0065894, -0.0000714, -0.0064914, -0.0006393, -0.0059501, 0.0064200
7: -0.0117850, 0.0000863, -0.0111989, -0.0001393, -0.0116456, 0.0112852
8: -0.0104798, 0.0022140, -0.0106104, 0.0009088, -0.0113887, 0.0128244
9: 0.9799623, 0.9978392, 0.9808776, 0.9971525, -0.0171902, 0.0169616

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0108670, upper bound: 0.0121119
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0120930
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0057226, 0.0051777, -0.0115625, 0.0102922
1: 0.0023423, 0.0136958, 0.0024777, 0.0126253, -0.0102831, 0.0112181
2: 0.0125382, 0.0300326, 0.0140326, 0.0305792, -0.0155119, 0.0138120
3: -0.0095456, -0.0011591, -0.0096271, -0.0019377, -0.0076079, 0.0084680
4: -0.0026604, 0.0023924, -0.0023095, 0.0022292, -0.0048896, 0.0047019
5: -0.0032551, 0.0087230, -0.0031822, 0.0074543, -0.0107095, 0.0119052
6: -0.0065894, -0.0000714, -0.0065304, -0.0006294, -0.0059600, 0.0064590
7: -0.0117850, 0.0000863, -0.0112415, -0.0000816, -0.0117034, 0.0113278
8: -0.0104798, 0.0022140, -0.0106450, 0.0008758, -0.0113556, 0.0128590
9: 0.9799623, 0.9978392, 0.9808148, 0.9972770, -0.0173146, 0.0170244

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0108670, upper bound: 0.0121119
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0120930
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0068992, 0.0047563, -0.0100268, 0.0117842
1: 0.0021074, 0.0124992, 0.0027820, 0.0138329, -0.0117255, 0.0097172
2: 0.0139343, 0.0305487, 0.0126169, 0.0299911, -0.0138303, 0.0154125
3: -0.0097542, -0.0020294, -0.0093943, -0.0010534, -0.0087007, 0.0073650
4: -0.0022694, 0.0025633, -0.0026949, 0.0020032, -0.0042726, 0.0052583
5: -0.0035617, 0.0073197, -0.0027867, 0.0088744, -0.0124361, 0.0101064
6: -0.0067548, -0.0006758, -0.0063192, -0.0000224, -0.0067324, 0.0056434
7: -0.0112094, 0.0003534, -0.0117971, -0.0004077, -0.0108017, 0.0121505
8: -0.0108022, 0.0007975, -0.0102907, 0.0023268, -0.0131290, 0.0110882
9: 0.9808887, 0.9981471, 0.9799096, 0.9968352, -0.0159466, 0.0182375

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110425, upper bound: 0.0114437
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106235, upper bound: 0.0114342
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0068766, 0.0048013, -0.0100811, 0.0117808
1: 0.0021074, 0.0124992, 0.0027177, 0.0138473, -0.0117399, 0.0097815
2: 0.0139343, 0.0305487, 0.0126341, 0.0300554, -0.0138978, 0.0154332
3: -0.0097542, -0.0020294, -0.0094224, -0.0010629, -0.0086912, 0.0073930
4: -0.0022694, 0.0025633, -0.0027126, 0.0020633, -0.0043327, 0.0052759
5: -0.0035617, 0.0073197, -0.0028694, 0.0088796, -0.0124413, 0.0101890
6: -0.0067548, -0.0006758, -0.0063635, -0.0000061, -0.0067487, 0.0056877
7: -0.0112094, 0.0003534, -0.0118321, -0.0003420, -0.0108674, 0.0121855
8: -0.0108022, 0.0007975, -0.0103321, 0.0023020, -0.0131042, 0.0111296
9: 0.9808887, 0.9981471, 0.9798547, 0.9969749, -0.0160863, 0.0182924

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110425, upper bound: 0.0114437
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106235, upper bound: 0.0114342
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0068992, 0.0047563, -0.0110793, 0.0113525
1: 0.0023423, 0.0136958, 0.0027820, 0.0138329, -0.0114906, 0.0109138
2: 0.0125382, 0.0300326, 0.0126169, 0.0299911, -0.0146251, 0.0145798
3: -0.0095456, -0.0011591, -0.0093943, -0.0010534, -0.0084922, 0.0082352
4: -0.0026604, 0.0023924, -0.0026949, 0.0020032, -0.0046636, 0.0050873
5: -0.0032551, 0.0087230, -0.0027867, 0.0088744, -0.0121295, 0.0115097
6: -0.0065894, -0.0000714, -0.0063192, -0.0000224, -0.0065670, 0.0062479
7: -0.0117850, 0.0000863, -0.0117971, -0.0004077, -0.0113773, 0.0118834
8: -0.0104798, 0.0022140, -0.0102907, 0.0023268, -0.0128067, 0.0125047
9: 0.9799623, 0.9978392, 0.9799096, 0.9968352, -0.0168729, 0.0179296

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0112879
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0112389
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0068766, 0.0048013, -0.0111333, 0.0113495
1: 0.0023423, 0.0136958, 0.0027177, 0.0138473, -0.0115050, 0.0109781
2: 0.0125382, 0.0300326, 0.0126341, 0.0300554, -0.0146936, 0.0145975
3: -0.0095456, -0.0011591, -0.0094224, -0.0010629, -0.0084827, 0.0082633
4: -0.0026604, 0.0023924, -0.0027126, 0.0020633, -0.0047237, 0.0051050
5: -0.0032551, 0.0087230, -0.0028694, 0.0088796, -0.0121347, 0.0115924
6: -0.0065894, -0.0000714, -0.0063635, -0.0000061, -0.0065833, 0.0062922
7: -0.0117850, 0.0000863, -0.0118321, -0.0003420, -0.0114429, 0.0119184
8: -0.0104798, 0.0022140, -0.0103321, 0.0023020, -0.0127819, 0.0125461
9: 0.9799623, 0.9978392, 0.9798547, 0.9969749, -0.0170126, 0.0179845

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0112879
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0112389
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0057548, 0.0051352, -0.0103375, 0.0106678
1: 0.0020438, 0.0125097, 0.0025355, 0.0126286, -0.0105847, 0.0099742
2: 0.0139545, 0.0306127, 0.0140131, 0.0305146, -0.0140090, 0.0140369
3: -0.0097794, -0.0020425, -0.0096019, -0.0019275, -0.0078519, 0.0075593
4: -0.0022855, 0.0026249, -0.0022970, 0.0021768, -0.0044623, 0.0049219
5: -0.0036424, 0.0073144, -0.0031136, 0.0074671, -0.0111096, 0.0104280
6: -0.0067977, -0.0006594, -0.0064914, -0.0006393, -0.0061585, 0.0058319
7: -0.0112578, 0.0004158, -0.0111989, -0.0001393, -0.0111185, 0.0116147
8: -0.0108384, 0.0007684, -0.0106104, 0.0009088, -0.0117473, 0.0113788
9: 0.9808157, 0.9982911, 0.9808776, 0.9971525, -0.0163368, 0.0174135

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110421, upper bound: 0.0115815
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106195, upper bound: 0.0115656
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0068992, 0.0047563, -0.0100225, 0.0118364
1: 0.0020438, 0.0125097, 0.0027820, 0.0138329, -0.0117891, 0.0097277
2: 0.0139545, 0.0306127, 0.0126169, 0.0299911, -0.0138494, 0.0154831
3: -0.0097794, -0.0020425, -0.0093943, -0.0010534, -0.0087260, 0.0073518
4: -0.0022855, 0.0026249, -0.0026949, 0.0020032, -0.0042887, 0.0053198
5: -0.0036424, 0.0073144, -0.0027867, 0.0088744, -0.0125168, 0.0101011
6: -0.0067977, -0.0006594, -0.0063192, -0.0000224, -0.0067753, 0.0056598
7: -0.0112578, 0.0004158, -0.0117971, -0.0004077, -0.0108501, 0.0122129
8: -0.0108384, 0.0007684, -0.0102907, 0.0023268, -0.0131652, 0.0110591
9: 0.9808157, 0.9982911, 0.9799096, 0.9968352, -0.0160195, 0.0183815

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110421, upper bound: 0.0115815
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106195, upper bound: 0.0115656
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0057548, 0.0051352, -0.0115167, 0.0103587
1: 0.0022689, 0.0137097, 0.0025355, 0.0126286, -0.0103596, 0.0111742
2: 0.0125547, 0.0300980, 0.0140131, 0.0305146, -0.0154632, 0.0138623
3: -0.0095757, -0.0011644, -0.0096019, -0.0019275, -0.0076482, 0.0084374
4: -0.0026819, 0.0024590, -0.0022970, 0.0021768, -0.0048587, 0.0047560
5: -0.0033472, 0.0087291, -0.0031136, 0.0074671, -0.0108143, 0.0118427
6: -0.0066386, -0.0000491, -0.0064914, -0.0006393, -0.0059993, 0.0064423
7: -0.0118299, 0.0001518, -0.0111989, -0.0001393, -0.0116906, 0.0113507
8: -0.0105247, 0.0021918, -0.0106104, 0.0009088, -0.0114335, 0.0128021
9: 0.9798938, 0.9979986, 0.9808776, 0.9971525, -0.0172587, 0.0171210

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0114553
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0113981
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0068992, 0.0047563, -0.0110853, 0.0114105
1: 0.0022689, 0.0137097, 0.0027820, 0.0138329, -0.0115640, 0.0109277
2: 0.0125547, 0.0300980, 0.0126169, 0.0299911, -0.0146430, 0.0146501
3: -0.0095757, -0.0011644, -0.0093943, -0.0010534, -0.0085223, 0.0082299
4: -0.0026819, 0.0024590, -0.0026949, 0.0020032, -0.0046851, 0.0051539
5: -0.0033472, 0.0087291, -0.0027867, 0.0088744, -0.0122216, 0.0115158
6: -0.0066386, -0.0000491, -0.0063192, -0.0000224, -0.0066162, 0.0062701
7: -0.0118299, 0.0001518, -0.0117971, -0.0004077, -0.0114222, 0.0119489
8: -0.0105247, 0.0021918, -0.0102907, 0.0023268, -0.0128515, 0.0124824
9: 0.9798938, 0.9979986, 0.9799096, 0.9968352, -0.0169414, 0.0180890

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0114553
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0113981
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0057226, 0.0051777, -0.0104095, 0.0106800
1: 0.0020438, 0.0125097, 0.0024777, 0.0126253, -0.0105815, 0.0100320
2: 0.0139545, 0.0306127, 0.0140326, 0.0305792, -0.0142149, 0.0141892
3: -0.0097794, -0.0020425, -0.0096271, -0.0019377, -0.0078417, 0.0075846
4: -0.0022855, 0.0026249, -0.0023095, 0.0022292, -0.0045147, 0.0049344
5: -0.0036424, 0.0073144, -0.0031822, 0.0074543, -0.0110968, 0.0104965
6: -0.0067977, -0.0006594, -0.0065304, -0.0006294, -0.0061684, 0.0058710
7: -0.0112578, 0.0004158, -0.0112415, -0.0000816, -0.0111762, 0.0116573
8: -0.0108384, 0.0007684, -0.0106450, 0.0008758, -0.0117142, 0.0114134
9: 0.9808157, 0.9982911, 0.9808148, 0.9972770, -0.0164613, 0.0174763

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110445, upper bound: 0.0115815
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106201, upper bound: 0.0115656
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0068766, 0.0048013, -0.0101014, 0.0118568
1: 0.0020438, 0.0125097, 0.0027177, 0.0138473, -0.0118035, 0.0097920
2: 0.0139545, 0.0306127, 0.0126341, 0.0300554, -0.0140668, 0.0156469
3: -0.0097794, -0.0020425, -0.0094224, -0.0010629, -0.0087165, 0.0073799
4: -0.0022855, 0.0026249, -0.0027126, 0.0020633, -0.0043488, 0.0053375
5: -0.0036424, 0.0073144, -0.0028694, 0.0088796, -0.0125220, 0.0101837
6: -0.0067977, -0.0006594, -0.0063635, -0.0000061, -0.0067917, 0.0057041
7: -0.0112578, 0.0004158, -0.0118321, -0.0003420, -0.0109158, 0.0122479
8: -0.0108384, 0.0007684, -0.0103321, 0.0023020, -0.0131405, 0.0111005
9: 0.9808157, 0.9982911, 0.9798547, 0.9969749, -0.0161592, 0.0184364

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0110445, upper bound: 0.0115815
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0106201, upper bound: 0.0115656
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0057226, 0.0051777, -0.0115905, 0.0103741
1: 0.0022689, 0.0137097, 0.0024777, 0.0126253, -0.0103564, 0.0112320
2: 0.0125547, 0.0300980, 0.0140326, 0.0305792, -0.0156782, 0.0140309
3: -0.0095757, -0.0011644, -0.0096271, -0.0019377, -0.0076379, 0.0084627
4: -0.0026819, 0.0024590, -0.0023095, 0.0022292, -0.0049111, 0.0047685
5: -0.0033472, 0.0087291, -0.0031822, 0.0074543, -0.0108015, 0.0119113
6: -0.0066386, -0.0000491, -0.0065304, -0.0006294, -0.0060092, 0.0064813
7: -0.0118299, 0.0001518, -0.0112415, -0.0000816, -0.0117483, 0.0113933
8: -0.0105247, 0.0021918, -0.0106450, 0.0008758, -0.0114005, 0.0128367
9: 0.9798938, 0.9979986, 0.9808148, 0.9972770, -0.0173832, 0.0171838

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105575, upper bound: 0.0114553
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099107, upper bound: 0.0113981
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0068766, 0.0048013, -0.0111657, 0.0114337
1: 0.0022689, 0.0137097, 0.0027177, 0.0138473, -0.0115784, 0.0109920
2: 0.0125547, 0.0300980, 0.0126341, 0.0300554, -0.0148606, 0.0148159
3: -0.0095757, -0.0011644, -0.0094224, -0.0010629, -0.0085128, 0.0082580
4: -0.0026819, 0.0024590, -0.0027126, 0.0020633, -0.0047452, 0.0051716
5: -0.0033472, 0.0087291, -0.0028694, 0.0088796, -0.0122268, 0.0115985
6: -0.0066386, -0.0000491, -0.0063635, -0.0000061, -0.0066325, 0.0063144
7: -0.0118299, 0.0001518, -0.0118321, -0.0003420, -0.0114879, 0.0119839
8: -0.0105247, 0.0021918, -0.0103321, 0.0023020, -0.0128267, 0.0125239
9: 0.9798938, 0.9979986, 0.9798547, 0.9969749, -0.0170811, 0.0181439

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105575, upper bound: 0.0114553
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099107, upper bound: 0.0113981
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0056457, 0.0052875, -0.0105834, 0.0105834
1: 0.0021074, 0.0124992, 0.0021074, 0.0124992, -0.0103918, 0.0103918
2: 0.0139343, 0.0305487, 0.0139343, 0.0305487, -0.0142443, 0.0142443
3: -0.0097542, -0.0020294, -0.0097542, -0.0020294, -0.0077248, 0.0077248
4: -0.0022694, 0.0025633, -0.0022694, 0.0025633, -0.0048328, 0.0048328
5: -0.0035617, 0.0073197, -0.0035617, 0.0073197, -0.0108814, 0.0108814
6: -0.0067548, -0.0006758, -0.0067548, -0.0006758, -0.0060790, 0.0060790
7: -0.0112094, 0.0003534, -0.0112094, 0.0003534, -0.0115628, 0.0115628
8: -0.0108022, 0.0007975, -0.0108022, 0.0007975, -0.0115997, 0.0115997
9: 0.9808887, 0.9981471, 0.9808887, 0.9981471, -0.0172585, 0.0172585

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120261, upper bound: 0.0127073
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127073
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0056172, 0.0053314, -0.0106362, 0.0105796
1: 0.0021074, 0.0124992, 0.0020438, 0.0125097, -0.0104023, 0.0104553
2: 0.0139343, 0.0305487, 0.0139545, 0.0306127, -0.0143146, 0.0142653
3: -0.0097542, -0.0020294, -0.0097794, -0.0020425, -0.0077116, 0.0077501
4: -0.0022694, 0.0025633, -0.0022855, 0.0026249, -0.0048943, 0.0048488
5: -0.0035617, 0.0073197, -0.0036424, 0.0073144, -0.0108761, 0.0109621
6: -0.0067548, -0.0006758, -0.0067977, -0.0006594, -0.0060954, 0.0061219
7: -0.0112094, 0.0003534, -0.0112578, 0.0004158, -0.0116252, 0.0116112
8: -0.0108022, 0.0007975, -0.0108384, 0.0007684, -0.0115706, 0.0116360
9: 0.9808887, 0.9981471, 0.9808157, 0.9982911, -0.0174025, 0.0173314

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120261, upper bound: 0.0127073
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127073
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0056457, 0.0052875, -0.0117519, 0.0102719
1: 0.0023423, 0.0136958, 0.0021074, 0.0124992, -0.0101569, 0.0115884
2: 0.0125382, 0.0300326, 0.0139343, 0.0305487, -0.0156782, 0.0140480
3: -0.0095456, -0.0011591, -0.0097542, -0.0020294, -0.0075162, 0.0085951
4: -0.0026604, 0.0023924, -0.0022694, 0.0025633, -0.0052237, 0.0046618
5: -0.0032551, 0.0087230, -0.0035617, 0.0073197, -0.0105748, 0.0122847
6: -0.0065894, -0.0000714, -0.0067548, -0.0006758, -0.0059136, 0.0066834
7: -0.0117850, 0.0000863, -0.0112094, 0.0003534, -0.0121384, 0.0112957
8: -0.0104798, 0.0022140, -0.0108022, 0.0007975, -0.0112774, 0.0130162
9: 0.9799623, 0.9978392, 0.9808887, 0.9981471, -0.0181848, 0.0169505

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115216, upper bound: 0.0122932
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0122905
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0056172, 0.0053314, -0.0118047, 0.0102682
1: 0.0023423, 0.0136958, 0.0020438, 0.0125097, -0.0101674, 0.0116519
2: 0.0125382, 0.0300326, 0.0139545, 0.0306127, -0.0157485, 0.0140691
3: -0.0095456, -0.0011591, -0.0097794, -0.0020425, -0.0075031, 0.0086203
4: -0.0026604, 0.0023924, -0.0022855, 0.0026249, -0.0052853, 0.0046779
5: -0.0032551, 0.0087230, -0.0036424, 0.0073144, -0.0105695, 0.0123654
6: -0.0065894, -0.0000714, -0.0067977, -0.0006594, -0.0059300, 0.0067264
7: -0.0117850, 0.0000863, -0.0112578, 0.0004158, -0.0122008, 0.0113441
8: -0.0104798, 0.0022140, -0.0108384, 0.0007684, -0.0112483, 0.0130524
9: 0.9799623, 0.9978392, 0.9808157, 0.9982911, -0.0183288, 0.0170235

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115216, upper bound: 0.0122932
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0122905
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0067875, 0.0049107, -0.0102719, 0.0117519
1: 0.0021074, 0.0124992, 0.0023423, 0.0136958, -0.0115884, 0.0101569
2: 0.0139343, 0.0305487, 0.0125382, 0.0300326, -0.0140480, 0.0156782
3: -0.0097542, -0.0020294, -0.0095456, -0.0011591, -0.0085951, 0.0075162
4: -0.0022694, 0.0025633, -0.0026604, 0.0023924, -0.0046618, 0.0052237
5: -0.0035617, 0.0073197, -0.0032551, 0.0087230, -0.0122847, 0.0105748
6: -0.0067548, -0.0006758, -0.0065894, -0.0000714, -0.0066834, 0.0059136
7: -0.0112094, 0.0003534, -0.0117850, 0.0000863, -0.0112957, 0.0121384
8: -0.0108022, 0.0007975, -0.0104798, 0.0022140, -0.0130162, 0.0112774
9: 0.9808887, 0.9981471, 0.9799623, 0.9978392, -0.0169505, 0.0181848

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0119331
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119326
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056457, 0.0052875, -0.0067659, 0.0049596, -0.0103292, 0.0117579
1: 0.0021074, 0.0124992, 0.0022689, 0.0137097, -0.0116023, 0.0102302
2: 0.0139343, 0.0305487, 0.0125547, 0.0300980, -0.0141171, 0.0156999
3: -0.0097542, -0.0020294, -0.0095757, -0.0011644, -0.0085897, 0.0075463
4: -0.0022694, 0.0025633, -0.0026819, 0.0024590, -0.0047284, 0.0052452
5: -0.0035617, 0.0073197, -0.0033472, 0.0087291, -0.0122908, 0.0106669
6: -0.0067548, -0.0006758, -0.0066386, -0.0000491, -0.0067057, 0.0059628
7: -0.0112094, 0.0003534, -0.0118299, 0.0001518, -0.0113612, 0.0121833
8: -0.0108022, 0.0007975, -0.0105247, 0.0021918, -0.0129940, 0.0113222
9: 0.9808887, 0.9981471, 0.9798938, 0.9979986, -0.0171099, 0.0182533

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0119331
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119326
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0067875, 0.0049107, -0.0113226, 0.0113226
1: 0.0023423, 0.0136958, 0.0023423, 0.0136958, -0.0113535, 0.0113535
2: 0.0125382, 0.0300326, 0.0125382, 0.0300326, -0.0148540, 0.0148540
3: -0.0095456, -0.0011591, -0.0095456, -0.0011591, -0.0083865, 0.0083865
4: -0.0026604, 0.0023924, -0.0026604, 0.0023924, -0.0050528, 0.0050528
5: -0.0032551, 0.0087230, -0.0032551, 0.0087230, -0.0119781, 0.0119781
6: -0.0065894, -0.0000714, -0.0065894, -0.0000714, -0.0065180, 0.0065180
7: -0.0117850, 0.0000863, -0.0117850, 0.0000863, -0.0118713, 0.0118713
8: -0.0104798, 0.0022140, -0.0104798, 0.0022140, -0.0126938, 0.0126938
9: 0.9799623, 0.9978392, 0.9799623, 0.9978392, -0.0178769, 0.0178769

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114811, upper bound: 0.0118308
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118247
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067875, 0.0049107, -0.0067659, 0.0049596, -0.0113811, 0.0113295
1: 0.0023423, 0.0136958, 0.0022689, 0.0137097, -0.0113674, 0.0114268
2: 0.0125382, 0.0300326, 0.0125547, 0.0300980, -0.0149235, 0.0148726
3: -0.0095456, -0.0011591, -0.0095757, -0.0011644, -0.0083812, 0.0084166
4: -0.0026604, 0.0023924, -0.0026819, 0.0024590, -0.0051194, 0.0050743
5: -0.0032551, 0.0087230, -0.0033472, 0.0087291, -0.0119842, 0.0120702
6: -0.0065894, -0.0000714, -0.0066386, -0.0000491, -0.0065403, 0.0065672
7: -0.0117850, 0.0000863, -0.0118299, 0.0001518, -0.0119368, 0.0119162
8: -0.0104798, 0.0022140, -0.0105247, 0.0021918, -0.0126716, 0.0127387
9: 0.9799623, 0.9978392, 0.9798938, 0.9979986, -0.0180362, 0.0179454

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114811, upper bound: 0.0118308
time: 3.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118247
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0056457, 0.0052875, -0.0105796, 0.0106362
1: 0.0020438, 0.0125097, 0.0021074, 0.0124992, -0.0104553, 0.0104023
2: 0.0139545, 0.0306127, 0.0139343, 0.0305487, -0.0142653, 0.0143146
3: -0.0097794, -0.0020425, -0.0097542, -0.0020294, -0.0077501, 0.0077116
4: -0.0022855, 0.0026249, -0.0022694, 0.0025633, -0.0048488, 0.0048943
5: -0.0036424, 0.0073144, -0.0035617, 0.0073197, -0.0109621, 0.0108761
6: -0.0067977, -0.0006594, -0.0067548, -0.0006758, -0.0061219, 0.0060954
7: -0.0112578, 0.0004158, -0.0112094, 0.0003534, -0.0116112, 0.0116252
8: -0.0108384, 0.0007684, -0.0108022, 0.0007975, -0.0116360, 0.0115706
9: 0.9808157, 0.9982911, 0.9808887, 0.9981471, -0.0173314, 0.0174025

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117746, upper bound: 0.0120406
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0067875, 0.0049107, -0.0102682, 0.0118047
1: 0.0020438, 0.0125097, 0.0023423, 0.0136958, -0.0116519, 0.0101674
2: 0.0139545, 0.0306127, 0.0125382, 0.0300326, -0.0140691, 0.0157485
3: -0.0097794, -0.0020425, -0.0095456, -0.0011591, -0.0086203, 0.0075031
4: -0.0022855, 0.0026249, -0.0026604, 0.0023924, -0.0046779, 0.0052853
5: -0.0036424, 0.0073144, -0.0032551, 0.0087230, -0.0123654, 0.0105695
6: -0.0067977, -0.0006594, -0.0065894, -0.0000714, -0.0067264, 0.0059300
7: -0.0112578, 0.0004158, -0.0117850, 0.0000863, -0.0113441, 0.0122008
8: -0.0108384, 0.0007684, -0.0104798, 0.0022140, -0.0130524, 0.0112483
9: 0.9808157, 0.9982911, 0.9799623, 0.9978392, -0.0170235, 0.0183288

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117746, upper bound: 0.0120406
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0056457, 0.0052875, -0.0117579, 0.0103292
1: 0.0022689, 0.0137097, 0.0021074, 0.0124992, -0.0102302, 0.0116023
2: 0.0125547, 0.0300980, 0.0139343, 0.0305487, -0.0156999, 0.0141171
3: -0.0095757, -0.0011644, -0.0097542, -0.0020294, -0.0075463, 0.0085897
4: -0.0026819, 0.0024590, -0.0022694, 0.0025633, -0.0052452, 0.0047284
5: -0.0033472, 0.0087291, -0.0035617, 0.0073197, -0.0106669, 0.0122908
6: -0.0066386, -0.0000491, -0.0067548, -0.0006758, -0.0059628, 0.0067057
7: -0.0118299, 0.0001518, -0.0112094, 0.0003534, -0.0121833, 0.0113612
8: -0.0105247, 0.0021918, -0.0108022, 0.0007975, -0.0113222, 0.0129940
9: 0.9798938, 0.9979986, 0.9808887, 0.9981471, -0.0182533, 0.0171099

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114808, upper bound: 0.0119663
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0067875, 0.0049107, -0.0113295, 0.0113811
1: 0.0022689, 0.0137097, 0.0023423, 0.0136958, -0.0114268, 0.0113674
2: 0.0125547, 0.0300980, 0.0125382, 0.0300326, -0.0148726, 0.0149235
3: -0.0095757, -0.0011644, -0.0095456, -0.0011591, -0.0084166, 0.0083812
4: -0.0026819, 0.0024590, -0.0026604, 0.0023924, -0.0050743, 0.0051194
5: -0.0033472, 0.0087291, -0.0032551, 0.0087230, -0.0120702, 0.0119842
6: -0.0066386, -0.0000491, -0.0065894, -0.0000714, -0.0065672, 0.0065403
7: -0.0118299, 0.0001518, -0.0117850, 0.0000863, -0.0119162, 0.0119368
8: -0.0105247, 0.0021918, -0.0104798, 0.0022140, -0.0127387, 0.0126716
9: 0.9798938, 0.9979986, 0.9799623, 0.9978392, -0.0179454, 0.0180362

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114808, upper bound: 0.0119663
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0056172, 0.0053314, -0.0106567, 0.0106567
1: 0.0020438, 0.0125097, 0.0020438, 0.0125097, -0.0104659, 0.0104659
2: 0.0139545, 0.0306127, 0.0139545, 0.0306127, -0.0144821, 0.0144821
3: -0.0097794, -0.0020425, -0.0097794, -0.0020425, -0.0077369, 0.0077369
4: -0.0022855, 0.0026249, -0.0022855, 0.0026249, -0.0049104, 0.0049104
5: -0.0036424, 0.0073144, -0.0036424, 0.0073144, -0.0109568, 0.0109568
6: -0.0067977, -0.0006594, -0.0067977, -0.0006594, -0.0061383, 0.0061383
7: -0.0112578, 0.0004158, -0.0112578, 0.0004158, -0.0116736, 0.0116736
8: -0.0108384, 0.0007684, -0.0108384, 0.0007684, -0.0116069, 0.0116069
9: 0.9808157, 0.9982911, 0.9808157, 0.9982911, -0.0174754, 0.0174754

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0120406
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115461, upper bound: 0.0120396
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056172, 0.0053314, -0.0067659, 0.0049596, -0.0103528, 0.0118365
1: 0.0020438, 0.0125097, 0.0022689, 0.0137097, -0.0116659, 0.0102408
2: 0.0139545, 0.0306127, 0.0125547, 0.0300980, -0.0142945, 0.0159226
3: -0.0097794, -0.0020425, -0.0095757, -0.0011644, -0.0086150, 0.0075332
4: -0.0022855, 0.0026249, -0.0026819, 0.0024590, -0.0047445, 0.0053068
5: -0.0036424, 0.0073144, -0.0033472, 0.0087291, -0.0123715, 0.0106616
6: -0.0067977, -0.0006594, -0.0066386, -0.0000491, -0.0067486, 0.0059791
7: -0.0112578, 0.0004158, -0.0118299, 0.0001518, -0.0114096, 0.0122457
8: -0.0108384, 0.0007684, -0.0105247, 0.0021918, -0.0130302, 0.0112931
9: 0.9808157, 0.9982911, 0.9798938, 0.9979986, -0.0171829, 0.0183973

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0120406
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115461, upper bound: 0.0120396
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0056172, 0.0053314, -0.0118365, 0.0103528
1: 0.0022689, 0.0137097, 0.0020438, 0.0125097, -0.0102408, 0.0116659
2: 0.0125547, 0.0300980, 0.0139545, 0.0306127, -0.0159226, 0.0142945
3: -0.0095757, -0.0011644, -0.0097794, -0.0020425, -0.0075332, 0.0086150
4: -0.0026819, 0.0024590, -0.0022855, 0.0026249, -0.0053068, 0.0047445
5: -0.0033472, 0.0087291, -0.0036424, 0.0073144, -0.0106616, 0.0123715
6: -0.0066386, -0.0000491, -0.0067977, -0.0006594, -0.0059791, 0.0067486
7: -0.0118299, 0.0001518, -0.0112578, 0.0004158, -0.0122457, 0.0114096
8: -0.0105247, 0.0021918, -0.0108384, 0.0007684, -0.0112931, 0.0130302
9: 0.9798938, 0.9979986, 0.9808157, 0.9982911, -0.0183973, 0.0171829

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114842, upper bound: 0.0119663
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111555, upper bound: 0.0119572
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067659, 0.0049596, -0.0067659, 0.0049596, -0.0114170, 0.0114170
1: 0.0022689, 0.0137097, 0.0022689, 0.0137097, -0.0114408, 0.0114408
2: 0.0125547, 0.0300980, 0.0125547, 0.0300980, -0.0150928, 0.0150928
3: -0.0095757, -0.0011644, -0.0095757, -0.0011644, -0.0084112, 0.0084112
4: -0.0026819, 0.0024590, -0.0026819, 0.0024590, -0.0051409, 0.0051409
5: -0.0033472, 0.0087291, -0.0033472, 0.0087291, -0.0120763, 0.0120763
6: -0.0066386, -0.0000491, -0.0066386, -0.0000491, -0.0065895, 0.0065895
7: -0.0118299, 0.0001518, -0.0118299, 0.0001518, -0.0119817, 0.0119817
8: -0.0105247, 0.0021918, -0.0105247, 0.0021918, -0.0127164, 0.0127164
9: 0.9798938, 0.9979986, 0.9798938, 0.9979986, -0.0181048, 0.0181048

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0114842, upper bound: 0.0119663
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111555, upper bound: 0.0119572
time: 1.00 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.46 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0128861, upper bound: 0.0115223
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115161
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0128861, upper bound: 0.0115223
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115161
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0121593, upper bound: 0.0107825
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114485, upper bound: 0.0106195
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0121593, upper bound: 0.0107825
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114485, upper bound: 0.0106195
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0124687, upper bound: 0.0102669
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0124687, upper bound: 0.0102669
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120518, upper bound: 0.0100636
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112642, upper bound: 0.0099079
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120518, upper bound: 0.0100636
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112642, upper bound: 0.0099079
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0124005, upper bound: 0.0102878
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0124005, upper bound: 0.0102878
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120126, upper bound: 0.0100939
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112389, upper bound: 0.0099244
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120126, upper bound: 0.0100939
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112389, upper bound: 0.0099244
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0124010, upper bound: 0.0102878
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0124010, upper bound: 0.0102878
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120153, upper bound: 0.0100939
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112417, upper bound: 0.0099244
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120153, upper bound: 0.0100939
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112417, upper bound: 0.0099244
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0118022, upper bound: 0.0126770
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0126770
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0118022, upper bound: 0.0126770
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0126770
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0108670, upper bound: 0.0121119
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0120930
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0108670, upper bound: 0.0121119
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0120930
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0110425, upper bound: 0.0114437
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0106235, upper bound: 0.0114342
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0110425, upper bound: 0.0114437
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0106235, upper bound: 0.0114342
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0112879
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0112389
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0112879
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0112389
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0110421, upper bound: 0.0115815
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0106195, upper bound: 0.0115656
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0110421, upper bound: 0.0115815
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0106195, upper bound: 0.0115656
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0114553
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0113981
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0105556, upper bound: 0.0114553
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0099079, upper bound: 0.0113981
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0110445, upper bound: 0.0115815
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0106201, upper bound: 0.0115656
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0110445, upper bound: 0.0115815
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0106201, upper bound: 0.0115656
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0105575, upper bound: 0.0114553
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0099107, upper bound: 0.0113981
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0105575, upper bound: 0.0114553
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0099107, upper bound: 0.0113981
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120261, upper bound: 0.0127073
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127073
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0120261, upper bound: 0.0127073
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127073
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115216, upper bound: 0.0122932
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0122905
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115216, upper bound: 0.0122932
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0122905
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0119331
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119326
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0119331
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119326
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114811, upper bound: 0.0118308
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118247
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114811, upper bound: 0.0118308
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118247
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0117746, upper bound: 0.0120406
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0117746, upper bound: 0.0120406
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114808, upper bound: 0.0119663
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114808, upper bound: 0.0119663
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0120406
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115461, upper bound: 0.0120396
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0117753, upper bound: 0.0120406
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0115461, upper bound: 0.0120396
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114842, upper bound: 0.0119663
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0111555, upper bound: 0.0119572
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0114842, upper bound: 0.0119663
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.46
Output dim: 9, lower bound: -0.0111555, upper bound: 0.0119572

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055504, 0.0051206, -0.0056457, 0.0052875, -0.0104212, 0.0103202
1: 0.0025582, 0.0124030, 0.0021074, 0.0124992, -0.0099409, 0.0102956
2: 0.0142554, 0.0304871, 0.0139343, 0.0305487, -0.0137416, 0.0139404
3: -0.0095944, -0.0021189, -0.0097542, -0.0020294, -0.0075651, 0.0076352
4: -0.0022307, 0.0021630, -0.0022694, 0.0025633, -0.0047941, 0.0044324
5: -0.0030963, 0.0072594, -0.0035617, 0.0073197, -0.0104160, 0.0108212
6: -0.0064831, -0.0007518, -0.0067548, -0.0006758, -0.0058074, 0.0060030
7: -0.0109915, -0.0001561, -0.0112094, 0.0003534, -0.0113449, 0.0110534
8: -0.0105999, 0.0006672, -0.0108022, 0.0007975, -0.0113974, 0.0114694
9: 0.9811344, 0.9971071, 0.9808887, 0.9981471, -0.0170127, 0.0162185

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115172
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115172
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051346, 0.0056594, -0.0055335, 0.0052801, -0.0100407, 0.0107155
1: 0.0021979, 0.0118107, 0.0021210, 0.0123430, -0.0101451, 0.0096897
2: 0.0146970, 0.0312175, 0.0140591, 0.0305349, -0.0135215, 0.0145145
3: -0.0099250, -0.0024929, -0.0097503, -0.0021347, -0.0077903, 0.0072573
4: -0.0020466, 0.0023222, -0.0022238, 0.0025562, -0.0046028, 0.0045460
5: -0.0034136, 0.0067091, -0.0035520, 0.0071753, -0.0105889, 0.0102611
6: -0.0066676, -0.0009800, -0.0067500, -0.0007512, -0.0059164, 0.0057700
7: -0.0109112, 0.0002076, -0.0110943, 0.0003432, -0.0112543, 0.0113019
8: -0.0110027, 0.0001760, -0.0107970, 0.0006623, -0.0116650, 0.0109729
9: 0.9811868, 0.9974339, 0.9810272, 0.9981212, -0.0169344, 0.0164067

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115172
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127301, upper bound: 0.0115172
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055504, 0.0051206, -0.0056172, 0.0053314, -0.0104734, 0.0103159
1: 0.0025582, 0.0124030, 0.0020438, 0.0125097, -0.0099515, 0.0103591
2: 0.0142554, 0.0304871, 0.0139545, 0.0306127, -0.0138122, 0.0139595
3: -0.0095944, -0.0021189, -0.0097794, -0.0020425, -0.0075519, 0.0076605
4: -0.0022307, 0.0021630, -0.0022855, 0.0026249, -0.0048556, 0.0044485
5: -0.0030963, 0.0072594, -0.0036424, 0.0073144, -0.0104107, 0.0109019
6: -0.0064831, -0.0007518, -0.0067977, -0.0006594, -0.0058237, 0.0060460
7: -0.0109915, -0.0001561, -0.0112578, 0.0004158, -0.0114073, 0.0111017
8: -0.0105999, 0.0006672, -0.0108384, 0.0007684, -0.0113684, 0.0115057
9: 0.9811344, 0.9971071, 0.9808157, 0.9982911, -0.0171567, 0.0162914

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128102, upper bound: 0.0115161
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128102, upper bound: 0.0115161
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051346, 0.0056594, -0.0055037, 0.0053240, -0.0100929, 0.0107115
1: 0.0021979, 0.0118107, 0.0020572, 0.0123542, -0.0101563, 0.0097534
2: 0.0146970, 0.0312175, 0.0140775, 0.0305987, -0.0135920, 0.0145343
3: -0.0099250, -0.0024929, -0.0097756, -0.0021483, -0.0077767, 0.0072827
4: -0.0020466, 0.0023222, -0.0022391, 0.0026178, -0.0046645, 0.0045613
5: -0.0034136, 0.0067091, -0.0036328, 0.0071699, -0.0105834, 0.0103419
6: -0.0066676, -0.0009800, -0.0067931, -0.0007341, -0.0059334, 0.0058131
7: -0.0109112, 0.0002076, -0.0111444, 0.0004055, -0.0113167, 0.0113520
8: -0.0110027, 0.0001760, -0.0108332, 0.0006332, -0.0116359, 0.0110092
9: 0.9811868, 0.9974339, 0.9809551, 0.9982659, -0.0170791, 0.0164788

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128102, upper bound: 0.0115161
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128102, upper bound: 0.0115161
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0067016, 0.0047409, -0.0056457, 0.0052875, -0.0115872, 0.0100044
1: 0.0028075, 0.0136001, 0.0021074, 0.0124992, -0.0096917, 0.0114927
2: 0.0128575, 0.0299621, 0.0139343, 0.0305487, -0.0151876, 0.0137721
3: -0.0093864, -0.0012384, -0.0097542, -0.0020294, -0.0073570, 0.0085158
4: -0.0026277, 0.0019903, -0.0022694, 0.0025633, -0.0051910, 0.0042597
5: -0.0027696, 0.0086598, -0.0035617, 0.0073197, -0.0100893, 0.0122215
6: -0.0063114, -0.0001448, -0.0067548, -0.0006758, -0.0056356, 0.0066100
7: -0.0115800, -0.0004245, -0.0112094, 0.0003534, -0.0119334, 0.0107849
8: -0.0102797, 0.0020920, -0.0108022, 0.0007975, -0.0110772, 0.0128942
9: 0.9801763, 0.9967926, 0.9808887, 0.9981471, -0.0179709, 0.0159039

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114485, upper bound: 0.0106235
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114485, upper bound: 0.0106235
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0067016, 0.0047409, -0.0056172, 0.0053314, -0.0116395, 0.0100001
1: 0.0028075, 0.0136001, 0.0020438, 0.0125097, -0.0097022, 0.0115563
2: 0.0128575, 0.0299621, 0.0139545, 0.0306127, -0.0152582, 0.0137913
3: -0.0093864, -0.0012384, -0.0097794, -0.0020425, -0.0073439, 0.0085410
4: -0.0026277, 0.0019903, -0.0022855, 0.0026249, -0.0052526, 0.0042758
5: -0.0027696, 0.0086598, -0.0036424, 0.0073144, -0.0100840, 0.0123022
6: -0.0063114, -0.0001448, -0.0067977, -0.0006594, -0.0056519, 0.0066530
7: -0.0115800, -0.0004245, -0.0112578, 0.0004158, -0.0119958, 0.0108333
8: -0.0102797, 0.0020920, -0.0108384, 0.0007684, -0.0110481, 0.0129305
9: 0.9801763, 0.9967926, 0.9808157, 0.9982911, -0.0181149, 0.0159768

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115656, upper bound: 0.0106195
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115656, upper bound: 0.0106195
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055504, 0.0051206, -0.0067875, 0.0049107, -0.0101077, 0.0114909
1: 0.0025582, 0.0124030, 0.0023423, 0.0136958, -0.0111375, 0.0100607
2: 0.0142554, 0.0304871, 0.0125382, 0.0300326, -0.0135683, 0.0153920
3: -0.0095944, -0.0021189, -0.0095456, -0.0011591, -0.0084353, 0.0074267
4: -0.0022307, 0.0021630, -0.0026604, 0.0023924, -0.0046231, 0.0048234
5: -0.0030963, 0.0072594, -0.0032551, 0.0087230, -0.0118193, 0.0105146
6: -0.0064831, -0.0007518, -0.0065894, -0.0000714, -0.0064118, 0.0058376
7: -0.0109915, -0.0001561, -0.0117850, 0.0000863, -0.0110778, 0.0116289
8: -0.0105999, 0.0006672, -0.0104798, 0.0022140, -0.0128139, 0.0111471
9: 0.9811344, 0.9971071, 0.9799623, 0.9978392, -0.0167048, 0.0171448

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051346, 0.0056594, -0.0066722, 0.0049029, -0.0097265, 0.0118865
1: 0.0021979, 0.0118107, 0.0023575, 0.0135430, -0.0113451, 0.0094532
2: 0.0146970, 0.0312175, 0.0126630, 0.0300180, -0.0133429, 0.0159707
3: -0.0099250, -0.0024929, -0.0095415, -0.0012647, -0.0086602, 0.0070485
4: -0.0020466, 0.0023222, -0.0026150, 0.0023856, -0.0044322, 0.0049371
5: -0.0034136, 0.0067091, -0.0032457, 0.0085836, -0.0119971, 0.0099548
6: -0.0066676, -0.0009800, -0.0065848, -0.0001474, -0.0065202, 0.0056047
7: -0.0109112, 0.0002076, -0.0116572, 0.0000762, -0.0109874, 0.0118648
8: -0.0110027, 0.0001760, -0.0104741, 0.0020769, -0.0130796, 0.0106501
9: 0.9811868, 0.9974339, 0.9801158, 0.9978139, -0.0166271, 0.0173181

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0121671, upper bound: 0.0102172
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055504, 0.0051206, -0.0067659, 0.0049596, -0.0101644, 0.0114951
1: 0.0025582, 0.0124030, 0.0022689, 0.0137097, -0.0111515, 0.0101340
2: 0.0142554, 0.0304871, 0.0125547, 0.0300980, -0.0136376, 0.0154137
3: -0.0095944, -0.0021189, -0.0095757, -0.0011644, -0.0084300, 0.0074568
4: -0.0022307, 0.0021630, -0.0026819, 0.0024590, -0.0046897, 0.0048449
5: -0.0030963, 0.0072594, -0.0033472, 0.0087291, -0.0118254, 0.0106066
6: -0.0064831, -0.0007518, -0.0066386, -0.0000491, -0.0064341, 0.0058868
7: -0.0109915, -0.0001561, -0.0118299, 0.0001518, -0.0111433, 0.0116739
8: -0.0105999, 0.0006672, -0.0105247, 0.0021918, -0.0127917, 0.0111919
9: 0.9811344, 0.9971071, 0.9798938, 0.9979986, -0.0168642, 0.0172133

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122817, upper bound: 0.0102172
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122817, upper bound: 0.0102172
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051346, 0.0056594, -0.0066506, 0.0049518, -0.0097832, 0.0118920
1: 0.0021979, 0.0118107, 0.0022838, 0.0135567, -0.0113588, 0.0095269
2: 0.0146970, 0.0312175, 0.0126798, 0.0300833, -0.0134126, 0.0159921
3: -0.0099250, -0.0024929, -0.0095716, -0.0012701, -0.0086548, 0.0070787
4: -0.0020466, 0.0023222, -0.0026350, 0.0024524, -0.0044990, 0.0049572
5: -0.0034136, 0.0067091, -0.0033378, 0.0085865, -0.0120001, 0.0100469
6: -0.0066676, -0.0009800, -0.0066339, -0.0001244, -0.0065432, 0.0056539
7: -0.0109112, 0.0002076, -0.0117028, 0.0001421, -0.0110533, 0.0119104
8: -0.0110027, 0.0001760, -0.0105190, 0.0020556, -0.0130583, 0.0106949
9: 0.9811868, 0.9974339, 0.9800444, 0.9979739, -0.0167871, 0.0173895

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122817, upper bound: 0.0102172
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122817, upper bound: 0.0102172
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0067016, 0.0047409, -0.0067875, 0.0049107, -0.0111566, 0.0110571
1: 0.0028075, 0.0136001, 0.0023423, 0.0136958, -0.0108883, 0.0112578
2: 0.0128575, 0.0299621, 0.0125382, 0.0300326, -0.0143565, 0.0145650
3: -0.0093864, -0.0012384, -0.0095456, -0.0011591, -0.0082273, 0.0083072
4: -0.0026277, 0.0019903, -0.0026604, 0.0023924, -0.0050201, 0.0046507
5: -0.0027696, 0.0086598, -0.0032551, 0.0087230, -0.0114926, 0.0119149
6: -0.0063114, -0.0001448, -0.0065894, -0.0000714, -0.0062400, 0.0064446
7: -0.0115800, -0.0004245, -0.0117850, 0.0000863, -0.0116663, 0.0113605
8: -0.0102797, 0.0020920, -0.0104798, 0.0022140, -0.0124937, 0.0125719
9: 0.9801763, 0.9967926, 0.9799623, 0.9978392, -0.0176629, 0.0168302

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112642, upper bound: 0.0099079
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112642, upper bound: 0.0099079
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0067016, 0.0047409, -0.0067659, 0.0049596, -0.0112147, 0.0110631
1: 0.0028075, 0.0136001, 0.0022689, 0.0137097, -0.0109022, 0.0113312
2: 0.0128575, 0.0299621, 0.0125547, 0.0300980, -0.0144269, 0.0145829
3: -0.0093864, -0.0012384, -0.0095757, -0.0011644, -0.0082220, 0.0083373
4: -0.0026277, 0.0019903, -0.0026819, 0.0024590, -0.0050867, 0.0046722
5: -0.0027696, 0.0086598, -0.0033472, 0.0087291, -0.0114987, 0.0120070
6: -0.0063114, -0.0001448, -0.0066386, -0.0000491, -0.0062623, 0.0064938
7: -0.0115800, -0.0004245, -0.0118299, 0.0001518, -0.0117318, 0.0114054
8: -0.0102797, 0.0020920, -0.0105247, 0.0021918, -0.0124714, 0.0126167
9: 0.9801763, 0.9967926, 0.9798938, 0.9979986, -0.0178223, 0.0168988

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113981, upper bound: 0.0099079
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0113981, upper bound: 0.0099079
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055196, 0.0051628, -0.0056457, 0.0052875, -0.0104095, 0.0103698
1: 0.0024996, 0.0123997, 0.0021074, 0.0124992, -0.0099996, 0.0102923
2: 0.0142752, 0.0305511, 0.0139343, 0.0305487, -0.0137601, 0.0140108
3: -0.0096194, -0.0021279, -0.0097542, -0.0020294, -0.0075900, 0.0076262
4: -0.0022421, 0.0022154, -0.0022694, 0.0025633, -0.0048054, 0.0044849
5: -0.0031646, 0.0072438, -0.0035617, 0.0073197, -0.0104843, 0.0108055
6: -0.0065222, -0.0007399, -0.0067548, -0.0006758, -0.0058464, 0.0060149
7: -0.0110316, -0.0000986, -0.0112094, 0.0003534, -0.0113850, 0.0111109
8: -0.0106343, 0.0006376, -0.0108022, 0.0007975, -0.0114318, 0.0114398
9: 0.9810725, 0.9972332, 0.9808887, 0.9981471, -0.0170746, 0.0163445

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051075, 0.0057052, -0.0055335, 0.0052801, -0.0100327, 0.0107726
1: 0.0021326, 0.0118123, 0.0021210, 0.0123430, -0.0102104, 0.0096913
2: 0.0147119, 0.0312822, 0.0140591, 0.0305349, -0.0135405, 0.0145873
3: -0.0099539, -0.0025051, -0.0097503, -0.0021347, -0.0078192, 0.0072452
4: -0.0020520, 0.0023748, -0.0022238, 0.0025562, -0.0046081, 0.0045986
5: -0.0034914, 0.0066938, -0.0035520, 0.0071753, -0.0106667, 0.0102458
6: -0.0067119, -0.0009692, -0.0067500, -0.0007512, -0.0059607, 0.0057809
7: -0.0109582, 0.0002690, -0.0110943, 0.0003432, -0.0113014, 0.0113633
8: -0.0110436, 0.0001518, -0.0107970, 0.0006623, -0.0117059, 0.0109488
9: 0.9811307, 0.9975685, 0.9810272, 0.9981212, -0.0169905, 0.0165413

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055196, 0.0051628, -0.0067875, 0.0049107, -0.0100960, 0.0115405
1: 0.0024996, 0.0123997, 0.0023423, 0.0136958, -0.0111962, 0.0100574
2: 0.0142752, 0.0305511, 0.0125382, 0.0300326, -0.0135868, 0.0154624
3: -0.0096194, -0.0021279, -0.0095456, -0.0011591, -0.0084603, 0.0074177
4: -0.0022421, 0.0022154, -0.0026604, 0.0023924, -0.0046345, 0.0048758
5: -0.0031646, 0.0072438, -0.0032551, 0.0087230, -0.0118876, 0.0104989
6: -0.0065222, -0.0007399, -0.0065894, -0.0000714, -0.0064508, 0.0058494
7: -0.0110316, -0.0000986, -0.0117850, 0.0000863, -0.0111179, 0.0116864
8: -0.0106343, 0.0006376, -0.0104798, 0.0022140, -0.0128483, 0.0111174
9: 0.9810725, 0.9972332, 0.9799623, 0.9978392, -0.0167667, 0.0172709

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051075, 0.0057052, -0.0066722, 0.0049029, -0.0097185, 0.0119435
1: 0.0021326, 0.0118123, 0.0023575, 0.0135430, -0.0114104, 0.0094548
2: 0.0147119, 0.0312822, 0.0126630, 0.0300180, -0.0133619, 0.0160434
3: -0.0099539, -0.0025051, -0.0095415, -0.0012647, -0.0086891, 0.0070364
4: -0.0020520, 0.0023748, -0.0026150, 0.0023856, -0.0044375, 0.0049898
5: -0.0034914, 0.0066938, -0.0032457, 0.0085836, -0.0120750, 0.0099396
6: -0.0067119, -0.0009692, -0.0065848, -0.0001474, -0.0065645, 0.0056156
7: -0.0109582, 0.0002690, -0.0116572, 0.0000762, -0.0110344, 0.0119262
8: -0.0110436, 0.0001518, -0.0104741, 0.0020769, -0.0131206, 0.0106259
9: 0.9811307, 0.9975685, 0.9801158, 0.9978139, -0.0166832, 0.0174527

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120930, upper bound: 0.0102331
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0066782, 0.0047854, -0.0056457, 0.0052875, -0.0115839, 0.0100581
1: 0.0027429, 0.0136122, 0.0021074, 0.0124992, -0.0097562, 0.0115048
2: 0.0128739, 0.0300257, 0.0139343, 0.0305487, -0.0152083, 0.0138397
3: -0.0094143, -0.0012462, -0.0097542, -0.0020294, -0.0073849, 0.0085079
4: -0.0026447, 0.0020503, -0.0022694, 0.0025633, -0.0052080, 0.0043198
5: -0.0028519, 0.0086673, -0.0035617, 0.0073197, -0.0101716, 0.0122291
6: -0.0063553, -0.0001301, -0.0067548, -0.0006758, -0.0056795, 0.0066247
7: -0.0116144, -0.0003591, -0.0112094, 0.0003534, -0.0119678, 0.0108503
8: -0.0103209, 0.0020648, -0.0108022, 0.0007975, -0.0111184, 0.0128670
9: 0.9801257, 0.9969323, 0.9808887, 0.9981471, -0.0180214, 0.0160436

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114342, upper bound: 0.0106498
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114342, upper bound: 0.0106498
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0066782, 0.0047854, -0.0067875, 0.0049107, -0.0111540, 0.0111106
1: 0.0027429, 0.0136122, 0.0023423, 0.0136958, -0.0109528, 0.0112699
2: 0.0128739, 0.0300257, 0.0125382, 0.0300326, -0.0143722, 0.0146339
3: -0.0094143, -0.0012462, -0.0095456, -0.0011591, -0.0082552, 0.0082994
4: -0.0026447, 0.0020503, -0.0026604, 0.0023924, -0.0050371, 0.0047107
5: -0.0028519, 0.0086673, -0.0032551, 0.0087230, -0.0115749, 0.0119224
6: -0.0063553, -0.0001301, -0.0065894, -0.0000714, -0.0062839, 0.0064592
7: -0.0116144, -0.0003591, -0.0117850, 0.0000863, -0.0117007, 0.0114259
8: -0.0103209, 0.0020648, -0.0104798, 0.0022140, -0.0125349, 0.0125447
9: 0.9801257, 0.9969323, 0.9799623, 0.9978392, -0.0177135, 0.0169699

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112389, upper bound: 0.0099244
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112389, upper bound: 0.0099244
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055196, 0.0051628, -0.0056172, 0.0053314, -0.0104835, 0.0103871
1: 0.0024996, 0.0123997, 0.0020438, 0.0125097, -0.0100101, 0.0103558
2: 0.0142752, 0.0305511, 0.0139545, 0.0306127, -0.0139643, 0.0141624
3: -0.0096194, -0.0021279, -0.0097794, -0.0020425, -0.0075769, 0.0076515
4: -0.0022421, 0.0022154, -0.0022855, 0.0026249, -0.0048670, 0.0045009
5: -0.0031646, 0.0072438, -0.0036424, 0.0073144, -0.0104790, 0.0108862
6: -0.0065222, -0.0007399, -0.0067977, -0.0006594, -0.0058628, 0.0060578
7: -0.0110316, -0.0000986, -0.0112578, 0.0004158, -0.0114474, 0.0111592
8: -0.0106343, 0.0006376, -0.0108384, 0.0007684, -0.0114027, 0.0114760
9: 0.9810725, 0.9972332, 0.9808157, 0.9982911, -0.0172186, 0.0164175

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051075, 0.0057052, -0.0055037, 0.0053240, -0.0101062, 0.0107939
1: 0.0021326, 0.0118123, 0.0020572, 0.0123542, -0.0102216, 0.0097551
2: 0.0147119, 0.0312822, 0.0140775, 0.0305987, -0.0137378, 0.0147477
3: -0.0099539, -0.0025051, -0.0097756, -0.0021483, -0.0078056, 0.0072705
4: -0.0020520, 0.0023748, -0.0022391, 0.0026178, -0.0046698, 0.0046139
5: -0.0034914, 0.0066938, -0.0036328, 0.0071699, -0.0106613, 0.0103266
6: -0.0067119, -0.0009692, -0.0067931, -0.0007341, -0.0059777, 0.0058240
7: -0.0109582, 0.0002690, -0.0111444, 0.0004055, -0.0113638, 0.0114135
8: -0.0110436, 0.0001518, -0.0108332, 0.0006332, -0.0116768, 0.0109851
9: 0.9811307, 0.9975685, 0.9809551, 0.9982659, -0.0171351, 0.0166134

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126770, upper bound: 0.0115443
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055196, 0.0051628, -0.0067659, 0.0049596, -0.0101776, 0.0115682
1: 0.0024996, 0.0123997, 0.0022689, 0.0137097, -0.0112101, 0.0101307
2: 0.0142752, 0.0305511, 0.0125547, 0.0300980, -0.0138060, 0.0156257
3: -0.0096194, -0.0021279, -0.0095757, -0.0011644, -0.0084550, 0.0074478
4: -0.0022421, 0.0022154, -0.0026819, 0.0024590, -0.0047011, 0.0048973
5: -0.0031646, 0.0072438, -0.0033472, 0.0087291, -0.0118937, 0.0105910
6: -0.0065222, -0.0007399, -0.0066386, -0.0000491, -0.0064731, 0.0058986
7: -0.0110316, -0.0000986, -0.0118299, 0.0001518, -0.0111834, 0.0117313
8: -0.0106343, 0.0006376, -0.0105247, 0.0021918, -0.0128260, 0.0111623
9: 0.9810725, 0.9972332, 0.9798938, 0.9979986, -0.0169261, 0.0173394

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051075, 0.0057052, -0.0066506, 0.0049518, -0.0097993, 0.0119763
1: 0.0021326, 0.0118123, 0.0022838, 0.0135567, -0.0114241, 0.0095285
2: 0.0147119, 0.0312822, 0.0126798, 0.0300833, -0.0135758, 0.0162119
3: -0.0099539, -0.0025051, -0.0095716, -0.0012701, -0.0086837, 0.0070665
4: -0.0020520, 0.0023748, -0.0026350, 0.0024524, -0.0045043, 0.0050098
5: -0.0034914, 0.0066938, -0.0033378, 0.0085865, -0.0120779, 0.0100316
6: -0.0067119, -0.0009692, -0.0066339, -0.0001244, -0.0065875, 0.0056647
7: -0.0109582, 0.0002690, -0.0117028, 0.0001421, -0.0111003, 0.0119718
8: -0.0110436, 0.0001518, -0.0105190, 0.0020556, -0.0130993, 0.0106708
9: 0.9811307, 0.9975685, 0.9800444, 0.9979739, -0.0168431, 0.0175241

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120969, upper bound: 0.0102331
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0066782, 0.0047854, -0.0056172, 0.0053314, -0.0116606, 0.0100779
1: 0.0027429, 0.0136122, 0.0020438, 0.0125097, -0.0097668, 0.0115683
2: 0.0128739, 0.0300257, 0.0139545, 0.0306127, -0.0154206, 0.0140085
3: -0.0094143, -0.0012462, -0.0097794, -0.0020425, -0.0073718, 0.0085332
4: -0.0026447, 0.0020503, -0.0022855, 0.0026249, -0.0052696, 0.0043358
5: -0.0028519, 0.0086673, -0.0036424, 0.0073144, -0.0101663, 0.0123098
6: -0.0063553, -0.0001301, -0.0067977, -0.0006594, -0.0056959, 0.0066676
7: -0.0116144, -0.0003591, -0.0112578, 0.0004158, -0.0120302, 0.0108987
8: -0.0103209, 0.0020648, -0.0108384, 0.0007684, -0.0110893, 0.0129033
9: 0.9801257, 0.9969323, 0.9808157, 0.9982911, -0.0181654, 0.0161166

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114391, upper bound: 0.0106498
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0114391, upper bound: 0.0106498
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0066782, 0.0047854, -0.0067659, 0.0049596, -0.0112384, 0.0111429
1: 0.0027429, 0.0136122, 0.0022689, 0.0137097, -0.0109668, 0.0113432
2: 0.0128739, 0.0300257, 0.0125547, 0.0300980, -0.0145929, 0.0148011
3: -0.0094143, -0.0012462, -0.0095757, -0.0011644, -0.0082499, 0.0083295
4: -0.0026447, 0.0020503, -0.0026819, 0.0024590, -0.0051037, 0.0047322
5: -0.0028519, 0.0086673, -0.0033472, 0.0087291, -0.0115810, 0.0120145
6: -0.0063553, -0.0001301, -0.0066386, -0.0000491, -0.0063062, 0.0065084
7: -0.0116144, -0.0003591, -0.0118299, 0.0001518, -0.0117662, 0.0114708
8: -0.0103209, 0.0020648, -0.0105247, 0.0021918, -0.0125127, 0.0125895
9: 0.9801257, 0.9969323, 0.9798938, 0.9979986, -0.0178729, 0.0170385

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112417, upper bound: 0.0099244
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0112417, upper bound: 0.0099244
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054424, 0.0052722, -0.0057548, 0.0051352, -0.0101518, 0.0105938
1: 0.0021319, 0.0122677, 0.0025355, 0.0126286, -0.0104967, 0.0097322
2: 0.0141765, 0.0305185, 0.0140131, 0.0305146, -0.0137652, 0.0139186
3: -0.0097467, -0.0022154, -0.0096019, -0.0019275, -0.0078192, 0.0073864
4: -0.0021996, 0.0025492, -0.0022970, 0.0021768, -0.0043764, 0.0048462
5: -0.0035442, 0.0071048, -0.0031136, 0.0074671, -0.0110114, 0.0102184
6: -0.0067464, -0.0007893, -0.0064914, -0.0006393, -0.0061071, 0.0057020
7: -0.0109997, 0.0003354, -0.0111989, -0.0001393, -0.0108604, 0.0115343
8: -0.0107919, 0.0005593, -0.0106104, 0.0009088, -0.0117008, 0.0111697
9: 0.9811490, 0.9981000, 0.9808776, 0.9971525, -0.0160035, 0.0172223

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0127301
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0127301
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050293, 0.0058145, -0.0056415, 0.0051283, -0.0097643, 0.0109942
1: 0.0017624, 0.0116866, 0.0025482, 0.0124695, -0.0107071, 0.0091383
2: 0.0146191, 0.0312481, 0.0141372, 0.0305023, -0.0135450, 0.0144978
3: -0.0100768, -0.0025913, -0.0095981, -0.0020323, -0.0080445, 0.0070068
4: -0.0020206, 0.0027045, -0.0022507, 0.0021698, -0.0041904, 0.0049551
5: -0.0038591, 0.0065567, -0.0031039, 0.0073209, -0.0111800, 0.0096607
6: -0.0069335, -0.0010101, -0.0064867, -0.0007155, -0.0062180, 0.0054766
7: -0.0109272, 0.0007051, -0.0110832, -0.0001488, -0.0107785, 0.0117884
8: -0.0111956, 0.0000703, -0.0106051, 0.0007749, -0.0119705, 0.0106754
9: 0.9811975, 0.9984303, 0.9810156, 0.9971273, -0.0159298, 0.0174146

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0127301
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115172, upper bound: 0.0127301
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054424, 0.0052722, -0.0057226, 0.0051777, -0.0102018, 0.0105840
1: 0.0021319, 0.0122677, 0.0024777, 0.0126253, -0.0104935, 0.0097900
2: 0.0141765, 0.0305185, 0.0140326, 0.0305792, -0.0138356, 0.0139375
3: -0.0097467, -0.0022154, -0.0096271, -0.0019377, -0.0078090, 0.0074117
4: -0.0021996, 0.0025492, -0.0023095, 0.0022292, -0.0044288, 0.0048587
5: -0.0035442, 0.0071048, -0.0031822, 0.0074543, -0.0109986, 0.0102870
6: -0.0067464, -0.0007893, -0.0065304, -0.0006294, -0.0061170, 0.0057410
7: -0.0109997, 0.0003354, -0.0112415, -0.0000816, -0.0109181, 0.0115768
8: -0.0107919, 0.0005593, -0.0106450, 0.0008758, -0.0116677, 0.0112043
9: 0.9811490, 0.9981000, 0.9808148, 0.9972770, -0.0161280, 0.0172852

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0126770
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0126770
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050293, 0.0058145, -0.0056090, 0.0051708, -0.0098142, 0.0109850
1: 0.0017624, 0.0116866, 0.0024898, 0.0124682, -0.0107058, 0.0091968
2: 0.0146191, 0.0312481, 0.0141552, 0.0305666, -0.0136156, 0.0145176
3: -0.0100768, -0.0025913, -0.0096232, -0.0020438, -0.0080330, 0.0070320
4: -0.0020206, 0.0027045, -0.0022625, 0.0022223, -0.0042429, 0.0049670
5: -0.0038591, 0.0065567, -0.0031725, 0.0073089, -0.0111680, 0.0097292
6: -0.0069335, -0.0010101, -0.0065258, -0.0007057, -0.0062278, 0.0055157
7: -0.0109272, 0.0007051, -0.0111266, -0.0000910, -0.0108362, 0.0118318
8: -0.0111956, 0.0000703, -0.0106397, 0.0007419, -0.0119375, 0.0107100
9: 0.9811975, 0.9984303, 0.9809545, 0.9972528, -0.0160553, 0.0174758

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0126770
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0126770
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0065856, 0.0048949, -0.0057548, 0.0051352, -0.0113191, 0.0102795
1: 0.0023694, 0.0134591, 0.0025355, 0.0126286, -0.0102592, 0.0109236
2: 0.0127785, 0.0300016, 0.0140131, 0.0305146, -0.0152173, 0.0137363
3: -0.0095377, -0.0013439, -0.0096019, -0.0019275, -0.0076102, 0.0082580
4: -0.0025918, 0.0023793, -0.0022970, 0.0021768, -0.0047686, 0.0046763
5: -0.0032385, 0.0085049, -0.0031136, 0.0074671, -0.0107056, 0.0116185
6: -0.0065814, -0.0001922, -0.0064914, -0.0006393, -0.0059421, 0.0062992
7: -0.0115663, 0.0000686, -0.0111989, -0.0001393, -0.0114270, 0.0112675
8: -0.0104689, 0.0019784, -0.0106104, 0.0009088, -0.0113777, 0.0125888
9: 0.9802308, 0.9977937, 0.9808776, 0.9971525, -0.0169217, 0.0169161

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0121671
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0121671
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0061763, 0.0054546, -0.0056415, 0.0051283, -0.0109319, 0.0106874
1: 0.0019701, 0.0129100, 0.0025482, 0.0124695, -0.0104993, 0.0103617
2: 0.0132310, 0.0307505, 0.0141372, 0.0305023, -0.0149970, 0.0142966
3: -0.0098815, -0.0017114, -0.0095981, -0.0020323, -0.0078493, 0.0078866
4: -0.0024143, 0.0025498, -0.0022507, 0.0021698, -0.0045841, 0.0048004
5: -0.0036012, 0.0079886, -0.0031039, 0.0073209, -0.0109221, 0.0110925
6: -0.0067921, -0.0004225, -0.0064867, -0.0007155, -0.0060766, 0.0060642
7: -0.0114083, 0.0004756, -0.0110832, -0.0001488, -0.0112596, 0.0115588
8: -0.0108988, 0.0014899, -0.0106051, 0.0007749, -0.0116737, 0.0120950
9: 0.9803578, 0.9981676, 0.9810156, 0.9971273, -0.0167695, 0.0171520

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0121671
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102172, upper bound: 0.0121671
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0065856, 0.0048949, -0.0057226, 0.0051777, -0.0113691, 0.0102697
1: 0.0023694, 0.0134591, 0.0024777, 0.0126253, -0.0102559, 0.0109814
2: 0.0127785, 0.0300016, 0.0140326, 0.0305792, -0.0152877, 0.0137553
3: -0.0095377, -0.0013439, -0.0096271, -0.0019377, -0.0076000, 0.0082832
4: -0.0025918, 0.0023793, -0.0023095, 0.0022292, -0.0048210, 0.0046887
5: -0.0032385, 0.0085049, -0.0031822, 0.0074543, -0.0106928, 0.0116871
6: -0.0065814, -0.0001922, -0.0065304, -0.0006294, -0.0059520, 0.0063382
7: -0.0115663, 0.0000686, -0.0112415, -0.0000816, -0.0114847, 0.0113101
8: -0.0104689, 0.0019784, -0.0106450, 0.0008758, -0.0113447, 0.0126234
9: 0.9802308, 0.9977937, 0.9808148, 0.9972770, -0.0170462, 0.0169789

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102331, upper bound: 0.0120930
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102331, upper bound: 0.0120930
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0061763, 0.0054546, -0.0056090, 0.0051708, -0.0109819, 0.0106782
1: 0.0019701, 0.0129100, 0.0024898, 0.0124682, -0.0104981, 0.0104202
2: 0.0132310, 0.0307505, 0.0141552, 0.0305666, -0.0150675, 0.0143165
3: -0.0098815, -0.0017114, -0.0096232, -0.0020438, -0.0078377, 0.0079118
4: -0.0024143, 0.0025498, -0.0022625, 0.0022223, -0.0046366, 0.0048123
5: -0.0036012, 0.0079886, -0.0031725, 0.0073089, -0.0109101, 0.0111611
6: -0.0067921, -0.0004225, -0.0065258, -0.0007057, -0.0060864, 0.0061033
7: -0.0114083, 0.0004756, -0.0111266, -0.0000910, -0.0113174, 0.0116023
8: -0.0108988, 0.0014899, -0.0106397, 0.0007419, -0.0116407, 0.0121296
9: 0.9803578, 0.9981676, 0.9809545, 0.9972528, -0.0168951, 0.0172131

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102331, upper bound: 0.0120930
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0102331, upper bound: 0.0120930
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054424, 0.0052722, -0.0056457, 0.0052875, -0.0103930, 0.0105616
1: 0.0021319, 0.0122677, 0.0021074, 0.0124992, -0.0103673, 0.0101603
2: 0.0141765, 0.0305185, 0.0139343, 0.0305487, -0.0140185, 0.0141935
3: -0.0097467, -0.0022154, -0.0097542, -0.0020294, -0.0077173, 0.0075387
4: -0.0021996, 0.0025492, -0.0022694, 0.0025633, -0.0047629, 0.0048186
5: -0.0035442, 0.0071048, -0.0035617, 0.0073197, -0.0108639, 0.0106666
6: -0.0067464, -0.0007893, -0.0067548, -0.0006758, -0.0060706, 0.0059654
7: -0.0109997, 0.0003354, -0.0112094, 0.0003534, -0.0113531, 0.0115448
8: -0.0107919, 0.0005593, -0.0108022, 0.0007975, -0.0115894, 0.0113615
9: 0.9811490, 0.9981000, 0.9808887, 0.9981471, -0.0169982, 0.0172113

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127625
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127625
time: 1.29 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050293, 0.0058145, -0.0055335, 0.0052801, -0.0100046, 0.0109583
1: 0.0017624, 0.0116866, 0.0021210, 0.0123430, -0.0105806, 0.0095655
2: 0.0146191, 0.0312481, 0.0140591, 0.0305349, -0.0138005, 0.0147566
3: -0.0100768, -0.0025913, -0.0097503, -0.0021347, -0.0079421, 0.0071590
4: -0.0020206, 0.0027045, -0.0022238, 0.0025562, -0.0045767, 0.0049282
5: -0.0038591, 0.0065567, -0.0035520, 0.0071753, -0.0110344, 0.0101087
6: -0.0069335, -0.0010101, -0.0067500, -0.0007512, -0.0061823, 0.0057399
7: -0.0109272, 0.0007051, -0.0110943, 0.0003432, -0.0112704, 0.0117995
8: -0.0111956, 0.0000703, -0.0107970, 0.0006623, -0.0118579, 0.0108673
9: 0.9811975, 0.9984303, 0.9810272, 0.9981212, -0.0169237, 0.0174031

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127625
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118960, upper bound: 0.0127625
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054424, 0.0052722, -0.0056172, 0.0053314, -0.0104458, 0.0105578
1: 0.0021319, 0.0122677, 0.0020438, 0.0125097, -0.0103778, 0.0102238
2: 0.0141765, 0.0305185, 0.0139545, 0.0306127, -0.0140888, 0.0142145
3: -0.0097467, -0.0022154, -0.0097794, -0.0020425, -0.0077042, 0.0075640
4: -0.0021996, 0.0025492, -0.0022855, 0.0026249, -0.0048245, 0.0048347
5: -0.0035442, 0.0071048, -0.0036424, 0.0073144, -0.0108586, 0.0107473
6: -0.0067464, -0.0007893, -0.0067977, -0.0006594, -0.0060870, 0.0060084
7: -0.0109997, 0.0003354, -0.0112578, 0.0004158, -0.0114155, 0.0115932
8: -0.0107919, 0.0005593, -0.0108384, 0.0007684, -0.0115604, 0.0113978
9: 0.9811490, 0.9981000, 0.9808157, 0.9982911, -0.0171422, 0.0172843

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119329, upper bound: 0.0127073
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119329, upper bound: 0.0127073
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050293, 0.0058145, -0.0055037, 0.0053240, -0.0100574, 0.0109563
1: 0.0017624, 0.0116866, 0.0020572, 0.0123542, -0.0105918, 0.0096293
2: 0.0146191, 0.0312481, 0.0140775, 0.0305987, -0.0138709, 0.0147773
3: -0.0100768, -0.0025913, -0.0097756, -0.0021483, -0.0079285, 0.0071844
4: -0.0020206, 0.0027045, -0.0022391, 0.0026178, -0.0046384, 0.0049436
5: -0.0038591, 0.0065567, -0.0036328, 0.0071699, -0.0110290, 0.0101895
6: -0.0069335, -0.0010101, -0.0067931, -0.0007341, -0.0061994, 0.0057830
7: -0.0109272, 0.0007051, -0.0111444, 0.0004055, -0.0113328, 0.0118496
8: -0.0111956, 0.0000703, -0.0108332, 0.0006332, -0.0118288, 0.0109035
9: 0.9811975, 0.9984303, 0.9809551, 0.9982659, -0.0170683, 0.0174751

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119329, upper bound: 0.0127073
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119329, upper bound: 0.0127073
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0065856, 0.0048949, -0.0056457, 0.0052875, -0.0115596, 0.0102494
1: 0.0023694, 0.0134591, 0.0021074, 0.0124992, -0.0101297, 0.0113517
2: 0.0127785, 0.0300016, 0.0139343, 0.0305487, -0.0154520, 0.0139907
3: -0.0095377, -0.0013439, -0.0097542, -0.0020294, -0.0075084, 0.0084103
4: -0.0025918, 0.0023793, -0.0022694, 0.0025633, -0.0051551, 0.0046487
5: -0.0032385, 0.0085049, -0.0035617, 0.0073197, -0.0105582, 0.0120667
6: -0.0065814, -0.0001922, -0.0067548, -0.0006758, -0.0059056, 0.0065626
7: -0.0115663, 0.0000686, -0.0112094, 0.0003534, -0.0119197, 0.0112780
8: -0.0104689, 0.0019784, -0.0108022, 0.0007975, -0.0112664, 0.0127806
9: 0.9802308, 0.9977937, 0.9808887, 0.9981471, -0.0179163, 0.0169051

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0123308
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0123308
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0061763, 0.0054546, -0.0055335, 0.0052801, -0.0111707, 0.0106519
1: 0.0019701, 0.0129100, 0.0021210, 0.0123430, -0.0103729, 0.0107889
2: 0.0132310, 0.0307505, 0.0140591, 0.0305349, -0.0152350, 0.0145498
3: -0.0098815, -0.0017114, -0.0097503, -0.0021347, -0.0077468, 0.0080389
4: -0.0024143, 0.0025498, -0.0022238, 0.0025562, -0.0049704, 0.0047735
5: -0.0036012, 0.0079886, -0.0035520, 0.0071753, -0.0107765, 0.0115405
6: -0.0067921, -0.0004225, -0.0067500, -0.0007512, -0.0060409, 0.0063275
7: -0.0114083, 0.0004756, -0.0110943, 0.0003432, -0.0117515, 0.0115699
8: -0.0108988, 0.0014899, -0.0107970, 0.0006623, -0.0115610, 0.0122869
9: 0.9803578, 0.9981676, 0.9810272, 0.9981212, -0.0177634, 0.0171404

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0123308
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0123308
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0065856, 0.0048949, -0.0056172, 0.0053314, -0.0116124, 0.0102456
1: 0.0023694, 0.0134591, 0.0020438, 0.0125097, -0.0101403, 0.0114153
2: 0.0127785, 0.0300016, 0.0139545, 0.0306127, -0.0155223, 0.0140117
3: -0.0095377, -0.0013439, -0.0097794, -0.0020425, -0.0074952, 0.0084355
4: -0.0025918, 0.0023793, -0.0022855, 0.0026249, -0.0052167, 0.0046648
5: -0.0032385, 0.0085049, -0.0036424, 0.0073144, -0.0105529, 0.0121474
6: -0.0065814, -0.0001922, -0.0067977, -0.0006594, -0.0059220, 0.0066055
7: -0.0115663, 0.0000686, -0.0112578, 0.0004158, -0.0119821, 0.0113264
8: -0.0104689, 0.0019784, -0.0108384, 0.0007684, -0.0112373, 0.0128168
9: 0.9802308, 0.9977937, 0.9808157, 0.9982911, -0.0180603, 0.0169780

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112469, upper bound: 0.0122905
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112469, upper bound: 0.0122905
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0061763, 0.0054546, -0.0055037, 0.0053240, -0.0112235, 0.0106499
1: 0.0019701, 0.0129100, 0.0020572, 0.0123542, -0.0103841, 0.0108527
2: 0.0132310, 0.0307505, 0.0140775, 0.0305987, -0.0153054, 0.0145705
3: -0.0098815, -0.0017114, -0.0097756, -0.0021483, -0.0077333, 0.0080642
4: -0.0024143, 0.0025498, -0.0022391, 0.0026178, -0.0050321, 0.0047889
5: -0.0036012, 0.0079886, -0.0036328, 0.0071699, -0.0107711, 0.0116214
6: -0.0067921, -0.0004225, -0.0067931, -0.0007341, -0.0060580, 0.0063706
7: -0.0114083, 0.0004756, -0.0111444, 0.0004055, -0.0118139, 0.0116201
8: -0.0108988, 0.0014899, -0.0108332, 0.0006332, -0.0115320, 0.0123231
9: 0.9803578, 0.9981676, 0.9809551, 0.9982659, -0.0179081, 0.0172125

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112469, upper bound: 0.0122905
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112469, upper bound: 0.0122905
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054424, 0.0052722, -0.0067875, 0.0049107, -0.0100815, 0.0117300
1: 0.0021319, 0.0122677, 0.0023423, 0.0136958, -0.0115639, 0.0099254
2: 0.0141765, 0.0305185, 0.0125382, 0.0300326, -0.0138223, 0.0156274
3: -0.0097467, -0.0022154, -0.0095456, -0.0011591, -0.0085876, 0.0073302
4: -0.0021996, 0.0025492, -0.0026604, 0.0023924, -0.0045920, 0.0052096
5: -0.0035442, 0.0071048, -0.0032551, 0.0087230, -0.0122672, 0.0103599
6: -0.0067464, -0.0007893, -0.0065894, -0.0000714, -0.0066750, 0.0058000
7: -0.0109997, 0.0003354, -0.0117850, 0.0000863, -0.0110860, 0.0121203
8: -0.0107919, 0.0005593, -0.0104798, 0.0022140, -0.0130059, 0.0110392
9: 0.9811490, 0.9981000, 0.9799623, 0.9978392, -0.0166903, 0.0181376

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119491
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119491
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050293, 0.0058145, -0.0066722, 0.0049029, -0.0096926, 0.0121281
1: 0.0017624, 0.0116866, 0.0023575, 0.0135430, -0.0117806, 0.0093291
2: 0.0146191, 0.0312481, 0.0126630, 0.0300180, -0.0136001, 0.0161912
3: -0.0100768, -0.0025913, -0.0095415, -0.0012647, -0.0088120, 0.0069502
4: -0.0020206, 0.0027045, -0.0026150, 0.0023856, -0.0044061, 0.0053194
5: -0.0038591, 0.0065567, -0.0032457, 0.0085836, -0.0124427, 0.0098025
6: -0.0069335, -0.0010101, -0.0065848, -0.0001474, -0.0067861, 0.0055746
7: -0.0109272, 0.0007051, -0.0116572, 0.0000762, -0.0110034, 0.0123624
8: -0.0111956, 0.0000703, -0.0104741, 0.0020769, -0.0132725, 0.0105444
9: 0.9811975, 0.9984303, 0.9801158, 0.9978139, -0.0166164, 0.0183145

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119491
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115489, upper bound: 0.0119491
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054424, 0.0052722, -0.0067659, 0.0049596, -0.0101387, 0.0117360
1: 0.0021319, 0.0122677, 0.0022689, 0.0137097, -0.0115778, 0.0099987
2: 0.0141765, 0.0305185, 0.0125547, 0.0300980, -0.0138913, 0.0156492
3: -0.0097467, -0.0022154, -0.0095757, -0.0011644, -0.0085823, 0.0073603
4: -0.0021996, 0.0025492, -0.0026819, 0.0024590, -0.0046586, 0.0052311
5: -0.0035442, 0.0071048, -0.0033472, 0.0087291, -0.0122733, 0.0104520
6: -0.0067464, -0.0007893, -0.0066386, -0.0000491, -0.0066973, 0.0058492
7: -0.0109997, 0.0003354, -0.0118299, 0.0001518, -0.0111515, 0.0121653
8: -0.0107919, 0.0005593, -0.0105247, 0.0021918, -0.0129837, 0.0110840
9: 0.9811490, 0.9981000, 0.9798938, 0.9979986, -0.0168496, 0.0182062

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116137, upper bound: 0.0119326
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116137, upper bound: 0.0119326
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050293, 0.0058145, -0.0066506, 0.0049518, -0.0097495, 0.0121344
1: 0.0017624, 0.0116866, 0.0022838, 0.0135567, -0.0117943, 0.0094028
2: 0.0146191, 0.0312481, 0.0126798, 0.0300833, -0.0136697, 0.0162133
3: -0.0100768, -0.0025913, -0.0095716, -0.0012701, -0.0088066, 0.0069803
4: -0.0020206, 0.0027045, -0.0026350, 0.0024524, -0.0044729, 0.0053395
5: -0.0038591, 0.0065567, -0.0033378, 0.0085865, -0.0124457, 0.0098946
6: -0.0069335, -0.0010101, -0.0066339, -0.0001244, -0.0068091, 0.0056238
7: -0.0109272, 0.0007051, -0.0117028, 0.0001421, -0.0110693, 0.0124080
8: -0.0111956, 0.0000703, -0.0105190, 0.0020556, -0.0132512, 0.0105893
9: 0.9811975, 0.9984303, 0.9800444, 0.9979739, -0.0167763, 0.0183858

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116137, upper bound: 0.0119326
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116137, upper bound: 0.0119326
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0065856, 0.0048949, -0.0067875, 0.0049107, -0.0111291, 0.0113002
1: 0.0023694, 0.0134591, 0.0023423, 0.0136958, -0.0113263, 0.0111168
2: 0.0127785, 0.0300016, 0.0125382, 0.0300326, -0.0146305, 0.0147955
3: -0.0095377, -0.0013439, -0.0095456, -0.0011591, -0.0083786, 0.0082017
4: -0.0025918, 0.0023793, -0.0026604, 0.0023924, -0.0049842, 0.0050396
5: -0.0032385, 0.0085049, -0.0032551, 0.0087230, -0.0119615, 0.0117600
6: -0.0065814, -0.0001922, -0.0065894, -0.0000714, -0.0065100, 0.0063972
7: -0.0115663, 0.0000686, -0.0117850, 0.0000863, -0.0116526, 0.0118536
8: -0.0104689, 0.0019784, -0.0104798, 0.0022140, -0.0126829, 0.0124582
9: 0.9802308, 0.9977937, 0.9799623, 0.9978392, -0.0176084, 0.0178314

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118448
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118448
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0061763, 0.0054546, -0.0066722, 0.0049029, -0.0107442, 0.0117092
1: 0.0019701, 0.0129100, 0.0023575, 0.0135430, -0.0115729, 0.0105525
2: 0.0132310, 0.0307505, 0.0126630, 0.0300180, -0.0144279, 0.0153230
3: -0.0098815, -0.0017114, -0.0095415, -0.0012647, -0.0086168, 0.0078300
4: -0.0024143, 0.0025498, -0.0026150, 0.0023856, -0.0047998, 0.0051647
5: -0.0036012, 0.0079886, -0.0032457, 0.0085836, -0.0121848, 0.0112343
6: -0.0067921, -0.0004225, -0.0065848, -0.0001474, -0.0066447, 0.0061623
7: -0.0114083, 0.0004756, -0.0116572, 0.0000762, -0.0114846, 0.0121328
8: -0.0108988, 0.0014899, -0.0104741, 0.0020769, -0.0129757, 0.0119640
9: 0.9803578, 0.9981676, 0.9801158, 0.9978139, -0.0174562, 0.0180518

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118448
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111484, upper bound: 0.0118448
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0065856, 0.0048949, -0.0067659, 0.0049596, -0.0111876, 0.0113071
1: 0.0023694, 0.0134591, 0.0022689, 0.0137097, -0.0113403, 0.0111902
2: 0.0127785, 0.0300016, 0.0125547, 0.0300980, -0.0146999, 0.0148141
3: -0.0095377, -0.0013439, -0.0095757, -0.0011644, -0.0083733, 0.0082318
4: -0.0025918, 0.0023793, -0.0026819, 0.0024590, -0.0050508, 0.0050612
5: -0.0032385, 0.0085049, -0.0033472, 0.0087291, -0.0119676, 0.0118521
6: -0.0065814, -0.0001922, -0.0066386, -0.0000491, -0.0065323, 0.0064464
7: -0.0115663, 0.0000686, -0.0118299, 0.0001518, -0.0117181, 0.0118985
8: -0.0104689, 0.0019784, -0.0105247, 0.0021918, -0.0126606, 0.0125031
9: 0.9802308, 0.9977937, 0.9798938, 0.9979986, -0.0177678, 0.0178999

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112029, upper bound: 0.0118247
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112029, upper bound: 0.0118247
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0061763, 0.0054546, -0.0066506, 0.0049518, -0.0108028, 0.0117156
1: 0.0019701, 0.0129100, 0.0022838, 0.0135567, -0.0115866, 0.0106262
2: 0.0132310, 0.0307505, 0.0126798, 0.0300833, -0.0144978, 0.0153425
3: -0.0098815, -0.0017114, -0.0095716, -0.0012701, -0.0086114, 0.0078602
4: -0.0024143, 0.0025498, -0.0026350, 0.0024524, -0.0048666, 0.0051848
5: -0.0036012, 0.0079886, -0.0033378, 0.0085865, -0.0121878, 0.0113264
6: -0.0067921, -0.0004225, -0.0066339, -0.0001244, -0.0066677, 0.0062114
7: -0.0114083, 0.0004756, -0.0117028, 0.0001421, -0.0115504, 0.0121784
8: -0.0108988, 0.0014899, -0.0105190, 0.0020556, -0.0129544, 0.0120089
9: 0.9803578, 0.9981676, 0.9800444, 0.9979739, -0.0176161, 0.0181231

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112029, upper bound: 0.0118247
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112029, upper bound: 0.0118247
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054163, 0.0053157, -0.0056457, 0.0052875, -0.0103880, 0.0106139
1: 0.0020679, 0.0122792, 0.0021074, 0.0124992, -0.0104312, 0.0101718
2: 0.0141969, 0.0305816, 0.0139343, 0.0305487, -0.0140407, 0.0142638
3: -0.0097719, -0.0022270, -0.0097542, -0.0020294, -0.0077425, 0.0075271
4: -0.0022156, 0.0026107, -0.0022694, 0.0025633, -0.0047789, 0.0048802
5: -0.0036250, 0.0071041, -0.0035617, 0.0073197, -0.0109447, 0.0106658
6: -0.0067895, -0.0007727, -0.0067548, -0.0006758, -0.0061137, 0.0059821
7: -0.0110457, 0.0003973, -0.0112094, 0.0003534, -0.0113991, 0.0116068
8: -0.0108279, 0.0005334, -0.0108022, 0.0007975, -0.0116254, 0.0113356
9: 0.9810768, 0.9982452, 0.9808887, 0.9981471, -0.0170704, 0.0173566

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118919, upper bound: 0.0128415
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118919, upper bound: 0.0128415
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050048, 0.0058623, -0.0055335, 0.0052801, -0.0100019, 0.0110181
1: 0.0016978, 0.0116952, 0.0021210, 0.0123430, -0.0106453, 0.0095741
2: 0.0146339, 0.0313131, 0.0140591, 0.0305349, -0.0138215, 0.0148305
3: -0.0101066, -0.0026006, -0.0097503, -0.0021347, -0.0079718, 0.0071497
4: -0.0020337, 0.0027676, -0.0022238, 0.0025562, -0.0045898, 0.0049914
5: -0.0039464, 0.0065545, -0.0035520, 0.0071753, -0.0111216, 0.0101065
6: -0.0069784, -0.0009943, -0.0067500, -0.0007512, -0.0062272, 0.0057558
7: -0.0109808, 0.0007648, -0.0110943, 0.0003432, -0.0113240, 0.0118591
8: -0.0112376, 0.0000493, -0.0107970, 0.0006623, -0.0118998, 0.0108462
9: 0.9811289, 0.9985755, 0.9810272, 0.9981212, -0.0169923, 0.0175483

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118919, upper bound: 0.0128415
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118919, upper bound: 0.0128415
time: 1.20 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054163, 0.0053157, -0.0067875, 0.0049107, -0.0100765, 0.0117824
1: 0.0020679, 0.0122792, 0.0023423, 0.0136958, -0.0116278, 0.0099369
2: 0.0141969, 0.0305816, 0.0125382, 0.0300326, -0.0138445, 0.0156977
3: -0.0097719, -0.0022270, -0.0095456, -0.0011591, -0.0086128, 0.0073186
4: -0.0022156, 0.0026107, -0.0026604, 0.0023924, -0.0046079, 0.0052711
5: -0.0036250, 0.0071041, -0.0032551, 0.0087230, -0.0123480, 0.0103592
6: -0.0067895, -0.0007727, -0.0065894, -0.0000714, -0.0067181, 0.0058167
7: -0.0110457, 0.0003973, -0.0117850, 0.0000863, -0.0111320, 0.0121823
8: -0.0108279, 0.0005334, -0.0104798, 0.0022140, -0.0130419, 0.0110133
9: 0.9810768, 0.9982452, 0.9799623, 0.9978392, -0.0167624, 0.0182829

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050048, 0.0058623, -0.0066722, 0.0049029, -0.0096899, 0.0121879
1: 0.0016978, 0.0116952, 0.0023575, 0.0135430, -0.0118452, 0.0093377
2: 0.0146339, 0.0313131, 0.0126630, 0.0300180, -0.0136211, 0.0162651
3: -0.0101066, -0.0026006, -0.0095415, -0.0012647, -0.0088418, 0.0069409
4: -0.0020337, 0.0027676, -0.0026150, 0.0023856, -0.0044192, 0.0053826
5: -0.0039464, 0.0065545, -0.0032457, 0.0085836, -0.0125299, 0.0098002
6: -0.0069784, -0.0009943, -0.0065848, -0.0001474, -0.0068310, 0.0055905
7: -0.0109808, 0.0007648, -0.0116572, 0.0000762, -0.0110570, 0.0124220
8: -0.0112376, 0.0000493, -0.0104741, 0.0020769, -0.0133145, 0.0105234
9: 0.9811289, 0.9985755, 0.9801158, 0.9978139, -0.0166850, 0.0184597

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0115443, upper bound: 0.0120396
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0065658, 0.0049433, -0.0056457, 0.0052875, -0.0115641, 0.0103058
1: 0.0022957, 0.0134698, 0.0021074, 0.0124992, -0.0102035, 0.0113624
2: 0.0127948, 0.0300662, 0.0139343, 0.0305487, -0.0154750, 0.0140601
3: -0.0095677, -0.0013485, -0.0097542, -0.0020294, -0.0075383, 0.0084057
4: -0.0026130, 0.0024460, -0.0022694, 0.0025633, -0.0051763, 0.0047154
5: -0.0033302, 0.0085114, -0.0035617, 0.0073197, -0.0106499, 0.0120731
6: -0.0066302, -0.0001710, -0.0067548, -0.0006758, -0.0059544, 0.0065838
7: -0.0116108, 0.0001344, -0.0112094, 0.0003534, -0.0119642, 0.0113439
8: -0.0105134, 0.0019554, -0.0108022, 0.0007975, -0.0113110, 0.0127576
9: 0.9801683, 0.9979540, 0.9808887, 0.9981471, -0.0179788, 0.0170653

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0124380
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0124380
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0061593, 0.0055015, -0.0055335, 0.0052801, -0.0111753, 0.0107112
1: 0.0019011, 0.0129232, 0.0021210, 0.0123430, -0.0104419, 0.0108022
2: 0.0132455, 0.0308153, 0.0140591, 0.0305349, -0.0152562, 0.0146193
3: -0.0099120, -0.0017147, -0.0097503, -0.0021347, -0.0077773, 0.0080356
4: -0.0024311, 0.0026171, -0.0022238, 0.0025562, -0.0049873, 0.0048408
5: -0.0036887, 0.0079964, -0.0035520, 0.0071753, -0.0108640, 0.0115484
6: -0.0068395, -0.0003997, -0.0067500, -0.0007512, -0.0060884, 0.0063504
7: -0.0114546, 0.0005399, -0.0110943, 0.0003432, -0.0117977, 0.0116342
8: -0.0109430, 0.0014759, -0.0107970, 0.0006623, -0.0116052, 0.0122728
9: 0.9802855, 0.9983330, 0.9810272, 0.9981212, -0.0178357, 0.0173059

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0124380
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0112159, upper bound: 0.0124380
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0065658, 0.0049433, -0.0067875, 0.0049107, -0.0111346, 0.0113582
1: 0.0022957, 0.0134698, 0.0023423, 0.0136958, -0.0114001, 0.0111275
2: 0.0127948, 0.0300662, 0.0125382, 0.0300326, -0.0146496, 0.0148653
3: -0.0095677, -0.0013485, -0.0095456, -0.0011591, -0.0084086, 0.0081971
4: -0.0026130, 0.0024460, -0.0026604, 0.0023924, -0.0050054, 0.0051064
5: -0.0033302, 0.0085114, -0.0032551, 0.0087230, -0.0120532, 0.0117665
6: -0.0066302, -0.0001710, -0.0065894, -0.0000714, -0.0065589, 0.0064183
7: -0.0116108, 0.0001344, -0.0117850, 0.0000863, -0.0116971, 0.0119194
8: -0.0105134, 0.0019554, -0.0104798, 0.0022140, -0.0127274, 0.0124353
9: 0.9801683, 0.9979540, 0.9799623, 0.9978392, -0.0176709, 0.0179916

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0061593, 0.0055015, -0.0066722, 0.0049029, -0.0107486, 0.0117677
1: 0.0019011, 0.0129232, 0.0023575, 0.0135430, -0.0116419, 0.0105657
2: 0.0132455, 0.0308153, 0.0126630, 0.0300180, -0.0144463, 0.0153944
3: -0.0099120, -0.0017147, -0.0095415, -0.0012647, -0.0086473, 0.0078268
4: -0.0024311, 0.0026171, -0.0026150, 0.0023856, -0.0048167, 0.0052320
5: -0.0036887, 0.0079964, -0.0032457, 0.0085836, -0.0122723, 0.0112422
6: -0.0068395, -0.0003997, -0.0065848, -0.0001474, -0.0066922, 0.0061851
7: -0.0114546, 0.0005399, -0.0116572, 0.0000762, -0.0115308, 0.0121971
8: -0.0109430, 0.0014759, -0.0104741, 0.0020769, -0.0130199, 0.0119500
9: 0.9802855, 0.9983330, 0.9801158, 0.9978139, -0.0175284, 0.0182173

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0111482, upper bound: 0.0119572
time: 1.05 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.16 + 597.54 = 600.70 seconds
