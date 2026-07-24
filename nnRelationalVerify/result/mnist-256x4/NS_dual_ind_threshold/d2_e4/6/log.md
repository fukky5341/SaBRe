## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.006503490000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142)
1: (0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415)
2: (-0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272)
3: (-0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832)
4: (0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747)
5: (-0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980)
6: (0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015)
7: (-0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0243378, 0.0243378)
8: (-0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058)
9: (-0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.01 + 3.65 = 5.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0069930, upper bound: 0.0069925

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067911, upper bound: 0.0066096
time: 2.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068719, upper bound: 0.0068708
time: 2.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.63 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.63
Output dim: 6, lower bound: -0.0067911, upper bound: 0.0066096
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.63
Output dim: 6, lower bound: -0.0068719, upper bound: 0.0068708

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0063121, 0.0083496, 0.0059328, 0.0084219, -0.0021098, 0.0024169
1: 0.0003081, 0.0042546, 0.0001555, 0.0043946, -0.0040865, 0.0040991
2: -0.0188016, 0.0144430, -0.0206734, 0.0156735, -0.0344751, 0.0351164
3: -0.0038025, 0.0019397, -0.0039124, 0.0035641, -0.0073666, 0.0058522
4: 0.0028869, 0.0166810, 0.0023537, 0.0171705, -0.0142836, 0.0143273
5: -0.0030099, 0.0014823, -0.0030829, 0.0037697, -0.0067795, 0.0045652
6: 0.9919966, 0.9984513, 0.9918507, 0.9999925, -0.0079959, 0.0066006
7: -0.0081571, 0.0168126, -0.0091223, 0.0176987, -0.0227398, 0.0228374
8: -0.0015672, 0.0062556, -0.0018696, 0.0065332, -0.0081004, 0.0081252
9: -0.0217527, -0.0042012, -0.0233251, -0.0035977, -0.0181550, 0.0191239

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066215, upper bound: 0.0065132
time: 2.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066755, upper bound: 0.0065140
time: 2.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0059957, 0.0084161, 0.0056663, 0.0084465, -0.0024508, 0.0027499
1: 0.0001731, 0.0043834, 0.0001520, 0.0044422, -0.0042690, 0.0042314
2: -0.0205234, 0.0155315, -0.0213087, 0.0157017, -0.0362251, 0.0368402
3: -0.0038998, 0.0034339, -0.0039150, 0.0041154, -0.0080152, 0.0073489
4: 0.0024152, 0.0171312, 0.0023415, 0.0173366, -0.0149214, 0.0147898
5: -0.0030771, 0.0035864, -0.0031077, 0.0045461, -0.0076232, 0.0066941
6: 0.9918676, 0.9998691, 0.9918474, 1.0005156, -0.0086480, 0.0080217
7: -0.0090109, 0.0176277, -0.0091444, 0.0179994, -0.0239107, 0.0227303
8: -0.0018347, 0.0065110, -0.0018765, 0.0066274, -0.0084621, 0.0083875
9: -0.0231991, -0.0036673, -0.0238589, -0.0035839, -0.0196152, 0.0201915

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066908, upper bound: 0.0067323
time: 2.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067334, upper bound: 0.0067330
time: 3.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 8.08 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 6, lower bound: -0.0066215, upper bound: 0.0065132
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 6, lower bound: -0.0066755, upper bound: 0.0065140
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 6, lower bound: -0.0066908, upper bound: 0.0067323
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 6, lower bound: -0.0067334, upper bound: 0.0067330

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0063184, 0.0083320, 0.0062556, 0.0083631, -0.0020447, 0.0020764
1: 0.0003203, 0.0042204, 0.0001986, 0.0042807, -0.0039604, 0.0040218
2: -0.0183448, 0.0143443, -0.0191513, 0.0153262, -0.0336710, 0.0334956
3: -0.0037937, 0.0015434, -0.0038814, 0.0022432, -0.0060370, 0.0054248
4: 0.0029297, 0.0165615, 0.0025042, 0.0167724, -0.0138428, 0.0140574
5: -0.0029920, 0.0009241, -0.0030235, 0.0019097, -0.0049017, 0.0039476
6: 0.9920084, 0.9980752, 0.9918919, 0.9987394, -0.0067310, 0.0061833
7: -0.0080796, 0.0165964, -0.0088499, 0.0169782, -0.0220063, 0.0223518
8: -0.0015429, 0.0061879, -0.0017843, 0.0063075, -0.0078504, 0.0079721
9: -0.0213689, -0.0042496, -0.0220465, -0.0037680, -0.0176009, 0.0177968

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064052, upper bound: 0.0062421
time: 2.81 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064956, upper bound: 0.0064048
time: 3.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0063223, 0.0083265, 0.0061315, 0.0083757, -0.0020535, 0.0021949
1: 0.0003278, 0.0042097, -0.0000417, 0.0043051, -0.0039773, 0.0042514
2: -0.0182018, 0.0142839, -0.0194773, 0.0172639, -0.0354658, 0.0337613
3: -0.0037883, 0.0014193, -0.0040545, 0.0025262, -0.0063145, 0.0054738
4: 0.0029558, 0.0165241, 0.0016645, 0.0168577, -0.0139019, 0.0148597
5: -0.0029864, 0.0007494, -0.0030362, 0.0023081, -0.0052945, 0.0037856
6: 0.9920157, 0.9979575, 0.9916621, 0.9990078, -0.0069921, 0.0062954
7: -0.0080323, 0.0165287, -0.0103698, 0.0171325, -0.0220663, 0.0236099
8: -0.0015281, 0.0061667, -0.0022605, 0.0063558, -0.0078839, 0.0084271
9: -0.0212488, -0.0042792, -0.0223203, -0.0028176, -0.0184312, 0.0180411

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064641, upper bound: 0.0062419
time: 3.05 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065576, upper bound: 0.0064060
time: 3.10 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0061780, 0.0083994, 0.0062537, 0.0083891, -0.0022111, 0.0021457
1: 0.0001855, 0.0043509, 0.0001949, 0.0043310, -0.0041455, 0.0041559
2: -0.0200888, 0.0154320, -0.0198225, 0.0153556, -0.0354444, 0.0352544
3: -0.0038909, 0.0030568, -0.0038841, 0.0028256, -0.0067165, 0.0069408
4: 0.0024583, 0.0170176, 0.0024915, 0.0169479, -0.0144896, 0.0145262
5: -0.0030601, 0.0030553, -0.0030497, 0.0027298, -0.0057899, 0.0061050
6: 0.9918794, 0.9995112, 0.9918884, 0.9992920, -0.0074126, 0.0076228
7: -0.0089328, 0.0174219, -0.0088729, 0.0172959, -0.0232043, 0.0222544
8: -0.0018102, 0.0064465, -0.0017915, 0.0064070, -0.0082173, 0.0082380
9: -0.0228340, -0.0037161, -0.0226103, -0.0037536, -0.0190804, 0.0188941

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064784, upper bound: 0.0064685
time: 2.82 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065603, upper bound: 0.0066123
time: 2.97 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0062527, 0.0083912, 0.0061295, 0.0083987, -0.0021460, 0.0022617
1: 0.0001930, 0.0043350, -0.0000456, 0.0043495, -0.0041565, 0.0043807
2: -0.0198772, 0.0153710, -0.0200709, 0.0172958, -0.0371730, 0.0354419
3: -0.0038854, 0.0028732, -0.0040573, 0.0030412, -0.0069266, 0.0069305
4: 0.0024848, 0.0169623, 0.0016507, 0.0170129, -0.0145281, 0.0153116
5: -0.0030518, 0.0027967, -0.0030594, 0.0030333, -0.0060852, 0.0058561
6: 0.9918866, 0.9993369, 0.9916583, 0.9994966, -0.0076100, 0.0076786
7: -0.0088850, 0.0173218, -0.0103948, 0.0174134, -0.0232531, 0.0236379
8: -0.0017953, 0.0064151, -0.0022683, 0.0064439, -0.0082391, 0.0086834
9: -0.0226563, -0.0037460, -0.0228189, -0.0028019, -0.0198543, 0.0190729

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065238, upper bound: 0.0064650
time: 3.13 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066135, upper bound: 0.0066127
time: 3.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 8.21 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0064052, upper bound: 0.0062421
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0064956, upper bound: 0.0064048
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0064641, upper bound: 0.0062419
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0065576, upper bound: 0.0064060
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0064784, upper bound: 0.0064685
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0065603, upper bound: 0.0066123
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0065238, upper bound: 0.0064650
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.21
Output dim: 6, lower bound: -0.0066135, upper bound: 0.0066127

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0063385, 0.0082859, 0.0061375, 0.0083614, -0.0020229, 0.0021484
1: 0.0003591, 0.0041311, -0.0000302, 0.0042773, -0.0039181, 0.0041613
2: -0.0171517, 0.0140311, -0.0191052, 0.0171713, -0.0343230, 0.0331362
3: -0.0037658, 0.0005080, -0.0040462, 0.0022032, -0.0059689, 0.0045542
4: 0.0030654, 0.0162495, 0.0017046, 0.0167604, -0.0136950, 0.0145449
5: -0.0029454, -0.0005339, -0.0030217, 0.0018532, -0.0047987, 0.0024878
6: 0.9920456, 0.9970928, 0.9916731, 0.9987014, -0.0066558, 0.0054198
7: -0.0078339, 0.0160316, -0.0102972, 0.0169563, -0.0217043, 0.0226775
8: -0.0014660, 0.0060109, -0.0022377, 0.0063006, -0.0077666, 0.0082486
9: -0.0203666, -0.0044033, -0.0220077, -0.0028630, -0.0175036, 0.0176044

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063337, upper bound: 0.0063072
time: 2.90 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063913, upper bound: 0.0063150
time: 2.78 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0062654, 0.0083547, 0.0062599, 0.0083730, -0.0021076, 0.0020947
1: 0.0002176, 0.0042643, 0.0002070, 0.0042998, -0.0040822, 0.0040573
2: -0.0189317, 0.0151731, -0.0194059, 0.0152582, -0.0341899, 0.0345790
3: -0.0038678, 0.0020527, -0.0038754, 0.0024641, -0.0063319, 0.0059281
4: 0.0025705, 0.0167150, 0.0025337, 0.0168390, -0.0142685, 0.0141814
5: -0.0030149, 0.0016414, -0.0030334, 0.0022208, -0.0052357, 0.0046748
6: 0.9919102, 0.9985586, 0.9919000, 0.9989489, -0.0070387, 0.0066586
7: -0.0087297, 0.0168742, -0.0087965, 0.0170987, -0.0228158, 0.0213900
8: -0.0017466, 0.0062749, -0.0017675, 0.0063452, -0.0080919, 0.0080425
9: -0.0218620, -0.0038431, -0.0222603, -0.0038014, -0.0180606, 0.0184172

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0065095
time: 3.25 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064562, upper bound: 0.0065112
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0062782, 0.0083045, 0.0061481, 0.0083614, -0.0020832, 0.0021563
1: 0.0002425, 0.0041671, -0.0000095, 0.0042774, -0.0040349, 0.0041766
2: -0.0176319, 0.0149722, -0.0191063, 0.0170045, -0.0346364, 0.0340785
3: -0.0038498, 0.0009248, -0.0040313, 0.0022042, -0.0060540, 0.0049561
4: 0.0026576, 0.0163751, 0.0017769, 0.0167607, -0.0141031, 0.0145982
5: -0.0029642, 0.0000530, -0.0030218, 0.0018546, -0.0048188, 0.0030747
6: 0.9919339, 0.9974883, 0.9916928, 0.9987022, -0.0067682, 0.0057955
7: -0.0085721, 0.0162589, -0.0101663, 0.0169568, -0.0223388, 0.0222571
8: -0.0016972, 0.0060822, -0.0021967, 0.0063008, -0.0079980, 0.0082789
9: -0.0207701, -0.0039417, -0.0220086, -0.0029449, -0.0178252, 0.0180670

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063371, upper bound: 0.0063599
time: 2.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064240, upper bound: 0.0063684
time: 3.17 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0062690, 0.0083474, 0.0061354, 0.0083834, -0.0021144, 0.0022120
1: 0.0002245, 0.0042503, -0.0000341, 0.0043200, -0.0040954, 0.0042844
2: -0.0187439, 0.0151168, -0.0196756, 0.0172032, -0.0359471, 0.0347924
3: -0.0038627, 0.0018897, -0.0040491, 0.0026982, -0.0065609, 0.0059388
4: 0.0025949, 0.0166659, 0.0016908, 0.0169095, -0.0143146, 0.0149751
5: -0.0030076, 0.0014119, -0.0030440, 0.0025503, -0.0055580, 0.0044559
6: 0.9919168, 0.9984040, 0.9916692, 0.9991709, -0.0072541, 0.0067348
7: -0.0086856, 0.0167853, -0.0103222, 0.0172263, -0.0228811, 0.0226575
8: -0.0017328, 0.0062471, -0.0022455, 0.0063852, -0.0081180, 0.0084926
9: -0.0217042, -0.0038707, -0.0224869, -0.0028474, -0.0188569, 0.0186162

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064334, upper bound: 0.0065086
time: 3.40 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065117, upper bound: 0.0065116
time: 3.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 8.68 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0063337, upper bound: 0.0063072
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0063913, upper bound: 0.0063150
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0065095
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0064562, upper bound: 0.0065112
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0063371, upper bound: 0.0063599
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0064240, upper bound: 0.0063684
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0064334, upper bound: 0.0065086
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.68
Output dim: 6, lower bound: -0.0065117, upper bound: 0.0065116

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0062741, 0.0083308, 0.0062849, 0.0083079, -0.0020338, 0.0020459
1: 0.0002345, 0.0042181, 0.0002555, 0.0041737, -0.0039392, 0.0039626
2: -0.0183145, 0.0150361, -0.0177212, 0.0148672, -0.0331817, 0.0327573
3: -0.0038555, 0.0015171, -0.0038404, 0.0010022, -0.0048577, 0.0053575
4: 0.0026299, 0.0165536, 0.0027031, 0.0163984, -0.0137686, 0.0138505
5: -0.0029908, 0.0008871, -0.0029677, 0.0001620, -0.0031529, 0.0038548
6: 0.9919264, 0.9980503, 0.9919464, 0.9975618, -0.0056354, 0.0061039
7: -0.0086223, 0.0165820, -0.0084898, 0.0163012, -0.0219408, 0.0207581
8: -0.0017130, 0.0061834, -0.0016715, 0.0060954, -0.0078084, 0.0078548
9: -0.0213435, -0.0039103, -0.0208450, -0.0039932, -0.0173503, 0.0169347

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0064765
time: 2.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0065091
time: 2.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0062738, 0.0083340, 0.0062136, 0.0083282, -0.0020543, 0.0021204
1: 0.0002340, 0.0042244, 0.0001173, 0.0042130, -0.0039790, 0.0041071
2: -0.0183981, 0.0150408, -0.0182458, 0.0159816, -0.0343797, 0.0332866
3: -0.0038559, 0.0015896, -0.0039400, 0.0014574, -0.0053134, 0.0055296
4: 0.0026278, 0.0165755, 0.0022202, 0.0165356, -0.0139078, 0.0143553
5: -0.0029941, 0.0009893, -0.0029882, 0.0008031, -0.0037972, 0.0039774
6: 0.9919258, 0.9981191, 0.9918142, 0.9979937, -0.0060679, 0.0063049
7: -0.0086260, 0.0166216, -0.0093640, 0.0165495, -0.0221962, 0.0216790
8: -0.0017141, 0.0061958, -0.0019453, 0.0061732, -0.0078873, 0.0081411
9: -0.0214137, -0.0039080, -0.0212858, -0.0034465, -0.0179672, 0.0173778

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063127, upper bound: 0.0064799
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0063127, upper bound: 0.0065112
time: 3.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0062778, 0.0083235, 0.0061606, 0.0083164, -0.0020386, 0.0021629
1: 0.0002417, 0.0042039, 0.0000146, 0.0041902, -0.0039485, 0.0041893
2: -0.0181239, 0.0149783, -0.0179418, 0.0168104, -0.0349344, 0.0329201
3: -0.0038504, 0.0013517, -0.0040140, 0.0011936, -0.0050440, 0.0053657
4: 0.0026549, 0.0165038, 0.0018610, 0.0164561, -0.0138012, 0.0146428
5: -0.0029834, 0.0006542, -0.0029763, 0.0004316, -0.0034150, 0.0036305
6: 0.9919332, 0.9978934, 0.9917158, 0.9977434, -0.0058102, 0.0061776
7: -0.0085770, 0.0164918, -0.0100141, 0.0164056, -0.0219824, 0.0220500
8: -0.0016988, 0.0061551, -0.0021490, 0.0061281, -0.0078269, 0.0083041
9: -0.0211834, -0.0039386, -0.0210303, -0.0030400, -0.0181434, 0.0170917

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062992, upper bound: 0.0062549
time: 2.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063816, upper bound: 0.0064698
time: 3.02 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0062773, 0.0083272, 0.0060806, 0.0083410, -0.0020637, 0.0022466
1: 0.0002408, 0.0042112, -0.0001403, 0.0042379, -0.0039971, 0.0043515
2: -0.0182215, 0.0149859, -0.0185784, 0.0180595, -0.0362810, 0.0335644
3: -0.0038510, 0.0014364, -0.0041256, 0.0017461, -0.0055971, 0.0055619
4: 0.0026516, 0.0165293, 0.0013197, 0.0166226, -0.0139710, 0.0152095
5: -0.0029872, 0.0007735, -0.0030011, 0.0012096, -0.0041968, 0.0037746
6: 0.9919323, 0.9979738, 0.9915676, 0.9982677, -0.0063354, 0.0064062
7: -0.0085829, 0.0165380, -0.0109939, 0.0167070, -0.0222614, 0.0231952
8: -0.0017006, 0.0061696, -0.0024560, 0.0062225, -0.0079232, 0.0086255
9: -0.0212654, -0.0039349, -0.0215652, -0.0024274, -0.0188380, 0.0176303

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064058, upper bound: 0.0062702
time: 3.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064773, upper bound: 0.0064773
time: 2.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.08 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0064765
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0065091
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0063127, upper bound: 0.0064799
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0063127, upper bound: 0.0065112
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0062992, upper bound: 0.0062549
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0063816, upper bound: 0.0064698
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0064058, upper bound: 0.0062702
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 8.08
Output dim: 6, lower bound: -0.0064773, upper bound: 0.0064773

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0061655, 0.0083003, 0.0062849, 0.0083079, -0.0021424, 0.0020154
1: 0.0000242, 0.0041590, 0.0002555, 0.0041737, -0.0041495, 0.0039036
2: -0.0175246, 0.0167327, -0.0177212, 0.0148672, -0.0323918, 0.0344539
3: -0.0040070, 0.0008317, -0.0038404, 0.0010022, -0.0050093, 0.0046721
4: 0.0018947, 0.0163471, 0.0027031, 0.0163984, -0.0145037, 0.0136440
5: -0.0029600, -0.0000781, -0.0029677, 0.0001620, -0.0031220, 0.0028895
6: 0.9917251, 0.9974001, 0.9919464, 0.9975618, -0.0058366, 0.0054537
7: -0.0099531, 0.0162081, -0.0084898, 0.0163012, -0.0231216, 0.0203195
8: -0.0021299, 0.0060662, -0.0016715, 0.0060954, -0.0082253, 0.0077377
9: -0.0206800, -0.0030782, -0.0208450, -0.0039932, -0.0166868, 0.0177669

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061154, upper bound: 0.0064511
time: 2.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0061154, upper bound: 0.0065091
time: 2.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0061649, 0.0083041, 0.0062136, 0.0083282, -0.0021632, 0.0020905
1: 0.0000230, 0.0041664, 0.0001173, 0.0042130, -0.0041900, 0.0040491
2: -0.0176232, 0.0167423, -0.0182458, 0.0159816, -0.0336049, 0.0349881
3: -0.0040079, 0.0009172, -0.0039400, 0.0014574, -0.0054654, 0.0048571
4: 0.0018905, 0.0163728, 0.0022202, 0.0165356, -0.0146451, 0.0141527
5: -0.0029639, 0.0000423, -0.0029882, 0.0008031, -0.0037670, 0.0030305
6: 0.9917239, 0.9974811, 0.9918142, 0.9979937, -0.0062698, 0.0056669
7: -0.0099607, 0.0162548, -0.0093640, 0.0165495, -0.0233793, 0.0212662
8: -0.0021323, 0.0060809, -0.0019453, 0.0061732, -0.0083055, 0.0080262
9: -0.0207627, -0.0030734, -0.0212858, -0.0034465, -0.0173162, 0.0182123

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061887, upper bound: 0.0064602
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0061887, upper bound: 0.0065120
time: 1.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.96 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.96
Output dim: 6, lower bound: -0.0061154, upper bound: 0.0064511
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 6, lower bound: -0.0061154, upper bound: 0.0065091
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.96
Output dim: 6, lower bound: -0.0061887, upper bound: 0.0064602
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.96
Output dim: 6, lower bound: -0.0061887, upper bound: 0.0065120

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0061655, 0.0083003, 0.0062956, 0.0082790, -0.0021135, 0.0020047
1: 0.0000242, 0.0041590, 0.0002762, 0.0041178, -0.0040936, 0.0038828
2: -0.0175246, 0.0167327, -0.0169737, 0.0147000, -0.0322246, 0.0337064
3: -0.0040070, 0.0008317, -0.0038255, 0.0003536, -0.0043607, 0.0046572
4: 0.0018947, 0.0163471, 0.0027756, 0.0162030, -0.0143083, 0.0135715
5: -0.0029600, -0.0000781, -0.0029385, -0.0007514, -0.0022087, 0.0028604
6: 0.9917251, 0.9974001, 0.9919661, 0.9969464, -0.0052212, 0.0054340
7: -0.0099531, 0.0162081, -0.0083586, 0.0159473, -0.0219321, 0.0201946
8: -0.0021299, 0.0060662, -0.0016304, 0.0059845, -0.0081144, 0.0076966
9: -0.0206800, -0.0030782, -0.0202171, -0.0040752, -0.0166048, 0.0171390

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 132

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060323, upper bound: 0.0062556
time: 3.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0060765, upper bound: 0.0064685
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0061649, 0.0083041, 0.0062242, 0.0082975, -0.0021326, 0.0020799
1: 0.0000230, 0.0041664, 0.0001378, 0.0041537, -0.0041307, 0.0040286
2: -0.0176232, 0.0167423, -0.0174530, 0.0158165, -0.0334397, 0.0341953
3: -0.0040079, 0.0009172, -0.0039252, 0.0007695, -0.0047774, 0.0048424
4: 0.0018905, 0.0163728, 0.0022917, 0.0163283, -0.0144378, 0.0140811
5: -0.0029639, 0.0000423, -0.0029572, -0.0001657, -0.0027982, 0.0029995
6: 0.9917239, 0.9974811, 0.9918337, 0.9973410, -0.0056171, 0.0056474
7: -0.0099607, 0.0162548, -0.0092344, 0.0161742, -0.0221761, 0.0211416
8: -0.0021323, 0.0060809, -0.0019047, 0.0060556, -0.0081879, 0.0079856
9: -0.0207627, -0.0030734, -0.0206198, -0.0035275, -0.0172352, 0.0175463

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 132

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061133, upper bound: 0.0062691
time: 3.49 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0061558, upper bound: 0.0064768
time: 2.58 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 8.36 seconds
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.36
Output dim: 6, lower bound: -0.0060323, upper bound: 0.0062556
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.36
Output dim: 6, lower bound: -0.0060765, upper bound: 0.0064685
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.36
Output dim: 6, lower bound: -0.0061133, upper bound: 0.0062691
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.36
Output dim: 6, lower bound: -0.0061558, upper bound: 0.0064768

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.66 + 145.92 = 151.58 seconds
