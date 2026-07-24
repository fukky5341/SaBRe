## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01341639


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9888137, 1.0114884, 0.9888137, 1.0114884, -0.0226747, 0.0226747)
1: (-0.0040513, 0.0010406, -0.0040513, 0.0010406, -0.0050919, 0.0050919)
2: (-0.0167521, 0.0114156, -0.0167521, 0.0114156, -0.0281677, 0.0281677)
3: (-0.0064690, 0.0058130, -0.0064690, 0.0058130, -0.0122820, 0.0122820)
4: (-0.0024854, 0.0027374, -0.0024854, 0.0027374, -0.0052227, 0.0052227)
5: (-0.0206217, 0.0133173, -0.0206217, 0.0133173, -0.0331142, 0.0331142)
6: (-0.0018392, 0.0091007, -0.0018392, 0.0091007, -0.0109399, 0.0109399)
7: (-0.0078963, 0.0143910, -0.0078963, 0.0143910, -0.0222872, 0.0222872)
8: (-0.0037167, 0.0080039, -0.0037167, 0.0080039, -0.0117206, 0.0117206)
9: (-0.0111448, 0.0024459, -0.0111448, 0.0024459, -0.0135907, 0.0135907)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.19 + 3.41 = 5.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0149071, upper bound: 0.0149071

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145435, upper bound: 0.0145898
time: 1.92 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145898, upper bound: 0.0145898
time: 2.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.41
Output dim: 0, lower bound: -0.0145435, upper bound: 0.0145898
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.41
Output dim: 0, lower bound: -0.0145898, upper bound: 0.0145898

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9896126, 1.0111626, 0.9890397, 1.0113969, -0.0217842, 0.0221230
1: -0.0038522, 0.0010071, -0.0039950, 0.0010312, -0.0048834, 0.0050021
2: -0.0164736, 0.0103608, -0.0166738, 0.0111173, -0.0275909, 0.0270345
3: -0.0059889, 0.0057324, -0.0063332, 0.0057903, -0.0117792, 0.0120656
4: -0.0024511, 0.0025332, -0.0024757, 0.0026796, -0.0051307, 0.0050089
5: -0.0203988, 0.0119906, -0.0205590, 0.0129421, -0.0325443, 0.0317492
6: -0.0015025, 0.0088450, -0.0017440, 0.0090288, -0.0105313, 0.0105890
7: -0.0070250, 0.0142446, -0.0076499, 0.0143498, -0.0213748, 0.0218945
8: -0.0032585, 0.0079270, -0.0035871, 0.0079823, -0.0112408, 0.0115141
9: -0.0110555, 0.0019146, -0.0111197, 0.0022956, -0.0133512, 0.0130343

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0141643
time: 2.19 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141316, upper bound: 0.0141643
time: 2.40 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9894298, 1.0120441, 0.9890221, 1.0114192, -0.0219893, 0.0230219
1: -0.0038978, 0.0010976, -0.0039994, 0.0010335, -0.0049312, 0.0050969
2: -0.0172269, 0.0106021, -0.0166929, 0.0111405, -0.0283673, 0.0272951
3: -0.0060988, 0.0059505, -0.0063438, 0.0057959, -0.0118946, 0.0122943
4: -0.0025438, 0.0025799, -0.0024781, 0.0026841, -0.0052279, 0.0050580
5: -0.0210016, 0.0122941, -0.0205744, 0.0129713, -0.0331895, 0.0321661
6: -0.0015796, 0.0095365, -0.0017514, 0.0090463, -0.0106259, 0.0112879
7: -0.0072244, 0.0146404, -0.0076691, 0.0143599, -0.0215843, 0.0223095
8: -0.0033634, 0.0081351, -0.0035972, 0.0079876, -0.0113509, 0.0117323
9: -0.0112969, 0.0020362, -0.0111258, 0.0023073, -0.0136042, 0.0131620

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0141648
time: 2.39 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141648, upper bound: 0.0141648
time: 2.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.00 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.00
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0141643
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.00
Output dim: 0, lower bound: -0.0141316, upper bound: 0.0141643
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.00
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0141648
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.00
Output dim: 0, lower bound: -0.0141648, upper bound: 0.0141648

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.9896163, 1.0111338, 0.9892142, 1.0101345, -0.0205181, 0.0219196
1: -0.0038513, 0.0010042, -0.0039515, 0.0009017, -0.0047530, 0.0049557
2: -0.0164491, 0.0103558, -0.0155951, 0.0108869, -0.0273360, 0.0259509
3: -0.0059866, 0.0057253, -0.0062284, 0.0054780, -0.0114646, 0.0119536
4: -0.0024481, 0.0025322, -0.0023429, 0.0026350, -0.0050831, 0.0048751
5: -0.0203793, 0.0119843, -0.0196959, 0.0126523, -0.0322349, 0.0308826
6: -0.0015009, 0.0088225, -0.0016704, 0.0080385, -0.0095394, 0.0104930
7: -0.0070209, 0.0142317, -0.0074596, 0.0137830, -0.0208039, 0.0216913
8: -0.0032564, 0.0079202, -0.0034871, 0.0076842, -0.0109406, 0.0114073
9: -0.0110477, 0.0019121, -0.0107740, 0.0021796, -0.0132273, 0.0126861

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0140758
time: 2.27 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0141643
time: 2.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9896935, 1.0104628, 0.9883508, 1.0097147, -0.0200212, 0.0221120
1: -0.0038320, 0.0009354, -0.0041666, 0.0008586, -0.0046907, 0.0051020
2: -0.0158755, 0.0102538, -0.0152364, 0.0120268, -0.0279024, 0.0254902
3: -0.0059402, 0.0055592, -0.0067472, 0.0053741, -0.0113143, 0.0123064
4: -0.0023774, 0.0025125, -0.0022987, 0.0028557, -0.0052331, 0.0048112
5: -0.0199203, 0.0118560, -0.0194089, 0.0140860, -0.0332706, 0.0310161
6: -0.0014684, 0.0082960, -0.0020344, 0.0077092, -0.0091776, 0.0103303
7: -0.0069367, 0.0139304, -0.0084011, 0.0135945, -0.0205312, 0.0223315
8: -0.0032121, 0.0077617, -0.0039822, 0.0075851, -0.0107972, 0.0117439
9: -0.0108639, 0.0018607, -0.0106591, 0.0027537, -0.0136176, 0.0125198

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138686, upper bound: 0.0130940
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139947, upper bound: 0.0140236
time: 2.23 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9894336, 1.0120174, 0.9891974, 1.0101552, -0.0207216, 0.0228200
1: -0.0038968, 0.0010948, -0.0039557, 0.0009038, -0.0048006, 0.0050505
2: -0.0172041, 0.0105971, -0.0156128, 0.0109090, -0.0281130, 0.0262099
3: -0.0060965, 0.0059439, -0.0062384, 0.0054831, -0.0115796, 0.0121823
4: -0.0025410, 0.0025789, -0.0023451, 0.0026393, -0.0051803, 0.0049240
5: -0.0209833, 0.0122878, -0.0197101, 0.0126801, -0.0328808, 0.0312986
6: -0.0015779, 0.0095156, -0.0016775, 0.0080548, -0.0096327, 0.0111931
7: -0.0072202, 0.0146284, -0.0074778, 0.0137923, -0.0210126, 0.0221063
8: -0.0033612, 0.0081288, -0.0034967, 0.0076891, -0.0110503, 0.0116255
9: -0.0112896, 0.0020336, -0.0107797, 0.0021907, -0.0134803, 0.0128134

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140758, upper bound: 0.0140758
time: 2.17 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140758, upper bound: 0.0141648
time: 2.26 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9895136, 1.0112830, 0.9883637, 1.0097380, -0.0202244, 0.0229193
1: -0.0038769, 0.0010195, -0.0041634, 0.0008610, -0.0047379, 0.0051829
2: -0.0165765, 0.0104915, -0.0152563, 0.0120099, -0.0285864, 0.0257478
3: -0.0060484, 0.0057622, -0.0067395, 0.0053799, -0.0114283, 0.0125017
4: -0.0024638, 0.0025585, -0.0023012, 0.0028524, -0.0053161, 0.0048597
5: -0.0204812, 0.0121550, -0.0194249, 0.0140647, -0.0337910, 0.0313125
6: -0.0015442, 0.0089395, -0.0020289, 0.0077275, -0.0092717, 0.0109684
7: -0.0071330, 0.0142987, -0.0083871, 0.0136050, -0.0207380, 0.0226858
8: -0.0033153, 0.0079554, -0.0039748, 0.0075906, -0.0109059, 0.0119302
9: -0.0110885, 0.0019804, -0.0106655, 0.0027452, -0.0138337, 0.0126460

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138892, upper bound: 0.0130940
time: 2.34 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140246, upper bound: 0.0140246
time: 2.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.90 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0140758
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0139989, upper bound: 0.0141643
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0138686, upper bound: 0.0130940
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0139947, upper bound: 0.0140236
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0140758, upper bound: 0.0140758
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0140758, upper bound: 0.0141648
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0138892, upper bound: 0.0130940
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.90
Output dim: 0, lower bound: -0.0140246, upper bound: 0.0140246

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9897838, 1.0099065, 0.9892142, 1.0101345, -0.0203506, 0.0206923
1: -0.0038096, 0.0008783, -0.0039515, 0.0009017, -0.0047112, 0.0048298
2: -0.0154003, 0.0101347, -0.0155951, 0.0108869, -0.0262872, 0.0257297
3: -0.0058860, 0.0054216, -0.0062284, 0.0054780, -0.0113640, 0.0116499
4: -0.0023189, 0.0024894, -0.0023429, 0.0026350, -0.0049539, 0.0048323
5: -0.0195401, 0.0117062, -0.0196959, 0.0126523, -0.0313979, 0.0306046
6: -0.0014303, 0.0078597, -0.0016704, 0.0080385, -0.0094688, 0.0095301
7: -0.0068383, 0.0136807, -0.0074596, 0.0137830, -0.0206213, 0.0211402
8: -0.0031604, 0.0076304, -0.0034871, 0.0076842, -0.0108445, 0.0111174
9: -0.0107116, 0.0018007, -0.0107740, 0.0021796, -0.0128912, 0.0125748

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0138011
time: 2.11 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138580, upper bound: 0.0139281
time: 2.32 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888838, 1.0094743, 0.9892142, 1.0101345, -0.0212507, 0.0202601
1: -0.0040338, 0.0008340, -0.0039515, 0.0009017, -0.0049355, 0.0047855
2: -0.0150309, 0.0113232, -0.0155951, 0.0108869, -0.0259178, 0.0269182
3: -0.0064269, 0.0053146, -0.0062284, 0.0054780, -0.0119049, 0.0115430
4: -0.0022734, 0.0027195, -0.0023429, 0.0026350, -0.0049085, 0.0050624
5: -0.0192445, 0.0132010, -0.0196959, 0.0126523, -0.0312048, 0.0319928
6: -0.0018097, 0.0075206, -0.0016704, 0.0080385, -0.0098482, 0.0091910
7: -0.0078200, 0.0134865, -0.0074596, 0.0137830, -0.0216029, 0.0209461
8: -0.0036766, 0.0075283, -0.0034871, 0.0076842, -0.0113608, 0.0110154
9: -0.0105933, 0.0023993, -0.0107740, 0.0021796, -0.0127729, 0.0131734

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0138889
time: 2.10 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138580, upper bound: 0.0140236
time: 2.41 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9898926, 1.0072087, 0.9884455, 1.0084561, -0.0185635, 0.0187631
1: -0.0037825, 0.0005322, -0.0041430, 0.0007295, -0.0045120, 0.0046753
2: -0.0128746, 0.0099910, -0.0141609, 0.0119018, -0.0247765, 0.0241519
3: -0.0058206, 0.0045869, -0.0066903, 0.0050627, -0.0108833, 0.0112772
4: -0.0019640, 0.0024616, -0.0021663, 0.0028315, -0.0047954, 0.0046279
5: -0.0172335, 0.0115255, -0.0185483, 0.0139288, -0.0303323, 0.0300738
6: -0.0013845, 0.0059149, -0.0019945, 0.0067219, -0.0081063, 0.0079093
7: -0.0067197, 0.0121660, -0.0082979, 0.0130294, -0.0197491, 0.0204638
8: -0.0030980, 0.0068338, -0.0039279, 0.0072879, -0.0103858, 0.0107617
9: -0.0097880, 0.0017284, -0.0103145, 0.0026908, -0.0124787, 0.0120429

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138686, upper bound: 0.0130368
time: 2.18 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138686, upper bound: 0.0130940
time: 2.20 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9897706, 1.0099703, 0.9883520, 1.0097069, -0.0199363, 0.0216183
1: -0.0038128, 0.0008848, -0.0041663, 0.0008578, -0.0046707, 0.0050512
2: -0.0154548, 0.0101521, -0.0152297, 0.0120253, -0.0274800, 0.0253818
3: -0.0058939, 0.0054373, -0.0067465, 0.0053722, -0.0112661, 0.0121839
4: -0.0023256, 0.0024928, -0.0022979, 0.0028554, -0.0051810, 0.0047907
5: -0.0195836, 0.0117281, -0.0194035, 0.0140841, -0.0320605, 0.0308886
6: -0.0014359, 0.0079097, -0.0020339, 0.0077030, -0.0091389, 0.0099435
7: -0.0068527, 0.0137093, -0.0083998, 0.0135910, -0.0204437, 0.0221091
8: -0.0031679, 0.0076454, -0.0039815, 0.0075832, -0.0107511, 0.0116269
9: -0.0107291, 0.0018095, -0.0106570, 0.0027529, -0.0134820, 0.0124665

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130431, upper bound: 0.0138889
time: 2.26 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130432, upper bound: 0.0140236
time: 2.38 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9896037, 1.0108711, 0.9891974, 1.0101552, -0.0205515, 0.0216737
1: -0.0038544, 0.0009772, -0.0039557, 0.0009038, -0.0047582, 0.0049329
2: -0.0162245, 0.0103725, -0.0156128, 0.0109090, -0.0271334, 0.0259853
3: -0.0059942, 0.0056602, -0.0062384, 0.0054831, -0.0114773, 0.0118986
4: -0.0024204, 0.0025355, -0.0023451, 0.0026393, -0.0050597, 0.0048805
5: -0.0201995, 0.0120053, -0.0197101, 0.0126801, -0.0320944, 0.0310173
6: -0.0015062, 0.0086163, -0.0016775, 0.0080548, -0.0095610, 0.0102938
7: -0.0070347, 0.0141137, -0.0074778, 0.0137923, -0.0208270, 0.0215915
8: -0.0032636, 0.0078581, -0.0034967, 0.0076891, -0.0109527, 0.0113548
9: -0.0109757, 0.0019205, -0.0107797, 0.0021907, -0.0131664, 0.0127002

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0138011
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139280, upper bound: 0.0139281
time: 2.20 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9888224, 1.0101365, 0.9891974, 1.0101552, -0.0213328, 0.0209391
1: -0.0040491, 0.0009019, -0.0039557, 0.0009038, -0.0049529, 0.0048575
2: -0.0155967, 0.0114041, -0.0156128, 0.0109090, -0.0265057, 0.0270170
3: -0.0064638, 0.0054785, -0.0062384, 0.0054831, -0.0119469, 0.0117169
4: -0.0023431, 0.0027351, -0.0023451, 0.0026393, -0.0049824, 0.0050802
5: -0.0196972, 0.0133028, -0.0197101, 0.0126801, -0.0316409, 0.0324480
6: -0.0018356, 0.0080400, -0.0016775, 0.0080548, -0.0098904, 0.0097175
7: -0.0078868, 0.0137839, -0.0074778, 0.0137923, -0.0216791, 0.0212617
8: -0.0037117, 0.0076846, -0.0034967, 0.0076891, -0.0114008, 0.0111813
9: -0.0107746, 0.0024401, -0.0107797, 0.0021907, -0.0129653, 0.0132198

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0138892
time: 2.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139280, upper bound: 0.0140246
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0075560, 0.9884589, 1.0084790, -0.0187572, 0.0190970
1: -0.0038250, 0.0006188, -0.0041397, 0.0007319, -0.0045569, 0.0047585
2: -0.0133332, 0.0102166, -0.0141805, 0.0118842, -0.0252174, 0.0243970
3: -0.0059233, 0.0047956, -0.0066823, 0.0050683, -0.0109916, 0.0114779
4: -0.0020527, 0.0025053, -0.0021687, 0.0028280, -0.0048808, 0.0046740
5: -0.0178103, 0.0118092, -0.0185640, 0.0139066, -0.0308764, 0.0303732
6: -0.0014565, 0.0060613, -0.0019888, 0.0067398, -0.0081963, 0.0080501
7: -0.0069059, 0.0125447, -0.0082833, 0.0130397, -0.0199456, 0.0208280
8: -0.0031959, 0.0070330, -0.0039202, 0.0072933, -0.0104892, 0.0109532
9: -0.0100190, 0.0018420, -0.0103208, 0.0026819, -0.0127008, 0.0121627

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138889, upper bound: 0.0130368
time: 2.40 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138889, upper bound: 0.0130393
time: 2.28 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9895927, 1.0107832, 0.9883649, 1.0097302, -0.0201375, 0.0224183
1: -0.0038572, 0.0009682, -0.0041631, 0.0008602, -0.0047174, 0.0051313
2: -0.0161494, 0.0103871, -0.0152496, 0.0120083, -0.0281577, 0.0256367
3: -0.0060009, 0.0056385, -0.0067388, 0.0053779, -0.0113788, 0.0123773
4: -0.0024112, 0.0025383, -0.0023004, 0.0028521, -0.0052632, 0.0048387
5: -0.0201395, 0.0120237, -0.0194195, 0.0140628, -0.0325716, 0.0311790
6: -0.0015109, 0.0085473, -0.0020284, 0.0077213, -0.0092322, 0.0105758
7: -0.0070468, 0.0140743, -0.0083858, 0.0136015, -0.0206482, 0.0224601
8: -0.0032700, 0.0078374, -0.0039742, 0.0075887, -0.0108587, 0.0118115
9: -0.0109517, 0.0019279, -0.0106634, 0.0027444, -0.0136961, 0.0125912

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0138892
time: 2.43 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0140246
time: 2.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.56 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0138011
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0138580, upper bound: 0.0139281
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0138889
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0138580, upper bound: 0.0140236
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0138686, upper bound: 0.0130368
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0138686, upper bound: 0.0130940
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0130431, upper bound: 0.0138889
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0130432, upper bound: 0.0140236
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0138011
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0139280, upper bound: 0.0139281
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0138892
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0139280, upper bound: 0.0140246
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0138889, upper bound: 0.0130368
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0138889, upper bound: 0.0130393
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0138892
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.56
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0140246

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9898740, 1.0086151, 0.9894263, 1.0070847, -0.0169204, 0.0191889
1: -0.0037871, 0.0007458, -0.0038987, 0.0005014, -0.0042885, 0.0045707
2: -0.0142968, 0.0100156, -0.0127110, 0.0106069, -0.0249037, 0.0223432
3: -0.0058318, 0.0051020, -0.0061009, 0.0045124, -0.0103441, 0.0110249
4: -0.0021830, 0.0024664, -0.0019323, 0.0025808, -0.0046882, 0.0043987
5: -0.0186571, 0.0115564, -0.0170276, 0.0123002, -0.0304653, 0.0274887
6: -0.0013923, 0.0068466, -0.0015811, 0.0058626, -0.0071326, 0.0084277
7: -0.0067400, 0.0131008, -0.0072284, 0.0120308, -0.0187707, 0.0200061
8: -0.0031086, 0.0073254, -0.0033655, 0.0067627, -0.0098713, 0.0105210
9: -0.0103581, 0.0017407, -0.0097056, 0.0020386, -0.0121996, 0.0114463

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127015, upper bound: 0.0132045
time: 2.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0135416
time: 2.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9897850, 1.0098994, 0.9892929, 1.0096668, -0.0198818, 0.0206065
1: -0.0038093, 0.0008776, -0.0039319, 0.0008537, -0.0046630, 0.0048094
2: -0.0153942, 0.0101332, -0.0151955, 0.0107829, -0.0261770, 0.0253287
3: -0.0058853, 0.0054198, -0.0061810, 0.0053623, -0.0112476, 0.0116008
4: -0.0023182, 0.0024892, -0.0022937, 0.0026149, -0.0049330, 0.0047829
5: -0.0195351, 0.0117043, -0.0193762, 0.0125215, -0.0312655, 0.0293946
6: -0.0014299, 0.0078540, -0.0016373, 0.0076716, -0.0091015, 0.0094913
7: -0.0068371, 0.0136774, -0.0073737, 0.0135730, -0.0204101, 0.0210511
8: -0.0031597, 0.0076287, -0.0034419, 0.0075738, -0.0107335, 0.0110706
9: -0.0107097, 0.0018000, -0.0106460, 0.0021272, -0.0128369, 0.0124460

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137946, upper bound: 0.0131128
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137946, upper bound: 0.0139577
time: 2.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889625, 1.0082163, 0.9894263, 1.0070847, -0.0178739, 0.0187900
1: -0.0040142, 0.0007049, -0.0038987, 0.0005014, -0.0045156, 0.0045574
2: -0.0139559, 0.0112191, -0.0127110, 0.0106069, -0.0245628, 0.0236023
3: -0.0063796, 0.0050033, -0.0061009, 0.0045124, -0.0108920, 0.0109928
4: -0.0021411, 0.0026993, -0.0019323, 0.0025808, -0.0046745, 0.0046316
5: -0.0183843, 0.0130701, -0.0170276, 0.0123002, -0.0303764, 0.0290554
6: -0.0017765, 0.0065337, -0.0015811, 0.0058626, -0.0075345, 0.0081148
7: -0.0077340, 0.0129217, -0.0072284, 0.0120308, -0.0197648, 0.0199477
8: -0.0036314, 0.0072312, -0.0033655, 0.0067627, -0.0103941, 0.0104903
9: -0.0102488, 0.0023469, -0.0097056, 0.0020386, -0.0121640, 0.0120525

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126412, upper bound: 0.0132926
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135728
time: 2.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888847, 1.0094664, 0.9892929, 1.0096668, -0.0207821, 0.0201735
1: -0.0040336, 0.0008332, -0.0039319, 0.0008537, -0.0048873, 0.0047650
2: -0.0150242, 0.0113218, -0.0151955, 0.0107829, -0.0258071, 0.0265173
3: -0.0064263, 0.0053127, -0.0061810, 0.0053623, -0.0117886, 0.0114937
4: -0.0022726, 0.0027192, -0.0022937, 0.0026149, -0.0048875, 0.0050129
5: -0.0192391, 0.0131993, -0.0193762, 0.0125215, -0.0310719, 0.0306328
6: -0.0018093, 0.0075144, -0.0016373, 0.0076716, -0.0094809, 0.0091516
7: -0.0078188, 0.0134830, -0.0073737, 0.0135730, -0.0213918, 0.0208567
8: -0.0036760, 0.0075264, -0.0034419, 0.0075738, -0.0112498, 0.0109683
9: -0.0105911, 0.0023986, -0.0106460, 0.0021272, -0.0127183, 0.0130447

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137318, upper bound: 0.0130940
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137318, upper bound: 0.0140236
time: 2.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9898926, 1.0072087, 0.9889625, 1.0082163, -0.0183237, 0.0181210
1: -0.0037825, 0.0005322, -0.0040142, 0.0007049, -0.0044874, 0.0045464
2: -0.0128746, 0.0099910, -0.0139559, 0.0112191, -0.0239286, 0.0239469
3: -0.0058206, 0.0045869, -0.0063796, 0.0050033, -0.0108239, 0.0109664
4: -0.0019640, 0.0024616, -0.0021411, 0.0026993, -0.0046633, 0.0046027
5: -0.0172335, 0.0115255, -0.0183843, 0.0130701, -0.0293195, 0.0299098
6: -0.0013845, 0.0059149, -0.0017765, 0.0065337, -0.0079181, 0.0076387
7: -0.0067197, 0.0121660, -0.0077340, 0.0129217, -0.0196413, 0.0198999
8: -0.0030980, 0.0068338, -0.0036314, 0.0072312, -0.0103292, 0.0104652
9: -0.0097880, 0.0017284, -0.0102488, 0.0023469, -0.0121349, 0.0119772

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135481, upper bound: 0.0125378
time: 2.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135576, upper bound: 0.0126875
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9898926, 1.0072087, 0.9889295, 1.0088203, -0.0189277, 0.0182792
1: -0.0037825, 0.0005322, -0.0040224, 0.0007669, -0.0045493, 0.0045547
2: -0.0128746, 0.0099910, -0.0144722, 0.0112629, -0.0241375, 0.0244632
3: -0.0058206, 0.0045869, -0.0063995, 0.0051528, -0.0109734, 0.0109863
4: -0.0019640, 0.0024616, -0.0022046, 0.0027078, -0.0046718, 0.0046663
5: -0.0172335, 0.0115255, -0.0187974, 0.0131252, -0.0295349, 0.0303229
6: -0.0013845, 0.0059149, -0.0017905, 0.0070076, -0.0083921, 0.0077053
7: -0.0067197, 0.0121660, -0.0077701, 0.0131929, -0.0199126, 0.0199361
8: -0.0030980, 0.0068338, -0.0036504, 0.0073739, -0.0104718, 0.0104842
9: -0.0097880, 0.0017284, -0.0104142, 0.0023689, -0.0121569, 0.0121426

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135481, upper bound: 0.0126033
time: 2.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135576, upper bound: 0.0127323
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9897706, 1.0099703, 0.9885970, 1.0069276, -0.0171570, 0.0213733
1: -0.0038128, 0.0008848, -0.0041053, 0.0004622, -0.0042751, 0.0049427
2: -0.0154548, 0.0101521, -0.0125036, 0.0117018, -0.0271566, 0.0226557
3: -0.0058939, 0.0054373, -0.0065993, 0.0044180, -0.0103119, 0.0119223
4: -0.0023256, 0.0024928, -0.0018922, 0.0027928, -0.0050698, 0.0043850
5: -0.0195836, 0.0117281, -0.0167668, 0.0136773, -0.0329451, 0.0281017
6: -0.0014359, 0.0079097, -0.0019306, 0.0057964, -0.0072323, 0.0098403
7: -0.0068527, 0.0137093, -0.0081327, 0.0118595, -0.0187122, 0.0216345
8: -0.0031679, 0.0076454, -0.0038410, 0.0066727, -0.0098406, 0.0113774
9: -0.0107291, 0.0018095, -0.0096011, 0.0025900, -0.0131926, 0.0114106

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0132926
time: 2.45 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
time: 2.41 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9897706, 1.0099703, 0.9884238, 1.0091914, -0.0194208, 0.0215465
1: -0.0038128, 0.0008848, -0.0041484, 0.0008049, -0.0046178, 0.0050333
2: -0.0154548, 0.0101521, -0.0147890, 0.0119305, -0.0273853, 0.0249411
3: -0.0058939, 0.0054373, -0.0067034, 0.0052446, -0.0111385, 0.0121407
4: -0.0023256, 0.0024928, -0.0022437, 0.0028370, -0.0051626, 0.0047364
5: -0.0195836, 0.0117281, -0.0190509, 0.0139649, -0.0319438, 0.0296699
6: -0.0014359, 0.0079097, -0.0020036, 0.0072985, -0.0087344, 0.0099133
7: -0.0068527, 0.0137093, -0.0083216, 0.0133594, -0.0202121, 0.0220308
8: -0.0031679, 0.0076454, -0.0039404, 0.0074615, -0.0106294, 0.0115858
9: -0.0107291, 0.0018095, -0.0105158, 0.0027052, -0.0134343, 0.0123253

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126949, upper bound: 0.0133805
time: 2.19 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0137192
time: 2.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9897105, 1.0096176, 0.9894083, 1.0070926, -0.0173821, 0.0202093
1: -0.0038278, 0.0008487, -0.0039031, 0.0005033, -0.0043311, 0.0047173
2: -0.0151534, 0.0102315, -0.0127214, 0.0106306, -0.0257840, 0.0229528
3: -0.0059300, 0.0053501, -0.0061117, 0.0045171, -0.0104471, 0.0113785
4: -0.0022885, 0.0025082, -0.0019343, 0.0025854, -0.0048385, 0.0044425
5: -0.0193425, 0.0118279, -0.0170407, 0.0123299, -0.0314423, 0.0280365
6: -0.0014612, 0.0076330, -0.0015886, 0.0058659, -0.0073272, 0.0092217
7: -0.0069183, 0.0135509, -0.0072479, 0.0120394, -0.0189576, 0.0206477
8: -0.0032024, 0.0075622, -0.0033757, 0.0067672, -0.0099696, 0.0108584
9: -0.0106325, 0.0018495, -0.0097108, 0.0020505, -0.0125909, 0.0115603

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127651, upper bound: 0.0132045
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0135416
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9896049, 1.0108631, 0.9892764, 1.0096875, -0.0200826, 0.0215867
1: -0.0038541, 0.0009764, -0.0039360, 0.0008559, -0.0047100, 0.0049124
2: -0.0162176, 0.0103709, -0.0152133, 0.0108046, -0.0270222, 0.0255842
3: -0.0059935, 0.0056582, -0.0061909, 0.0053674, -0.0113609, 0.0118492
4: -0.0024196, 0.0025352, -0.0022959, 0.0026191, -0.0050387, 0.0048310
5: -0.0201940, 0.0120033, -0.0193904, 0.0125488, -0.0319611, 0.0297752
6: -0.0015057, 0.0086100, -0.0016442, 0.0076880, -0.0091937, 0.0102542
7: -0.0070334, 0.0141101, -0.0073916, 0.0135824, -0.0206158, 0.0215017
8: -0.0032629, 0.0078562, -0.0034513, 0.0075787, -0.0108416, 0.0113076
9: -0.0109735, 0.0019197, -0.0106517, 0.0021381, -0.0131117, 0.0125714

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138422, upper bound: 0.0131128
time: 2.13 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0138422, upper bound: 0.0139577
time: 2.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889289, 1.0088717, 0.9894083, 1.0070926, -0.0181637, 0.0194634
1: -0.0040226, 0.0007722, -0.0039031, 0.0005033, -0.0045259, 0.0046594
2: -0.0145161, 0.0112636, -0.0127214, 0.0106306, -0.0251467, 0.0239850
3: -0.0063998, 0.0051655, -0.0061117, 0.0045171, -0.0109169, 0.0112389
4: -0.0022101, 0.0027079, -0.0019343, 0.0025854, -0.0047791, 0.0046422
5: -0.0188326, 0.0131261, -0.0170407, 0.0123299, -0.0310565, 0.0294685
6: -0.0017907, 0.0070480, -0.0015886, 0.0058659, -0.0076567, 0.0086366
7: -0.0077708, 0.0132161, -0.0072479, 0.0120394, -0.0198101, 0.0203943
8: -0.0036507, 0.0073860, -0.0033757, 0.0067672, -0.0104179, 0.0107252
9: -0.0104283, 0.0023693, -0.0097108, 0.0020505, -0.0124364, 0.0120801

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127255, upper bound: 0.0132927
time: 2.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135736
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888235, 1.0101287, 0.9892764, 1.0096875, -0.0208640, 0.0208523
1: -0.0040488, 0.0009011, -0.0039360, 0.0008559, -0.0049047, 0.0048371
2: -0.0155902, 0.0114026, -0.0152133, 0.0108046, -0.0263948, 0.0266159
3: -0.0064631, 0.0054766, -0.0061909, 0.0053674, -0.0118305, 0.0116675
4: -0.0023423, 0.0027348, -0.0022959, 0.0026191, -0.0049614, 0.0050307
5: -0.0196920, 0.0133009, -0.0193904, 0.0125488, -0.0315075, 0.0311924
6: -0.0018351, 0.0080340, -0.0016442, 0.0076880, -0.0095230, 0.0096782
7: -0.0078855, 0.0137804, -0.0073916, 0.0135824, -0.0214679, 0.0211721
8: -0.0037111, 0.0076828, -0.0034513, 0.0075787, -0.0112898, 0.0111342
9: -0.0107725, 0.0024393, -0.0106517, 0.0021381, -0.0129106, 0.0130910

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137988, upper bound: 0.0130940
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137988, upper bound: 0.0140246
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0075560, 0.9889625, 1.0082163, -0.0184945, 0.0184934
1: -0.0038250, 0.0006188, -0.0040142, 0.0007049, -0.0045299, 0.0046330
2: -0.0133332, 0.0102166, -0.0139559, 0.0112191, -0.0244203, 0.0241725
3: -0.0059233, 0.0047956, -0.0063796, 0.0050033, -0.0109266, 0.0111752
4: -0.0020527, 0.0025053, -0.0021411, 0.0026993, -0.0047521, 0.0046464
5: -0.0178103, 0.0118092, -0.0183843, 0.0130701, -0.0299142, 0.0301935
6: -0.0014565, 0.0060613, -0.0017765, 0.0065337, -0.0079901, 0.0077956
7: -0.0069059, 0.0125447, -0.0077340, 0.0129217, -0.0198276, 0.0202787
8: -0.0031959, 0.0070330, -0.0036314, 0.0072312, -0.0104272, 0.0106644
9: -0.0100190, 0.0018420, -0.0102488, 0.0023469, -0.0123659, 0.0120908

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135620, upper bound: 0.0125378
time: 2.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135728, upper bound: 0.0126875
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0075560, 0.9889295, 1.0088203, -0.0190985, 0.0186265
1: -0.0038250, 0.0006188, -0.0040224, 0.0007669, -0.0045919, 0.0046412
2: -0.0133332, 0.0102166, -0.0144722, 0.0112629, -0.0245961, 0.0246887
3: -0.0059233, 0.0047956, -0.0063995, 0.0051528, -0.0110761, 0.0111951
4: -0.0020527, 0.0025053, -0.0022046, 0.0027078, -0.0047605, 0.0047099
5: -0.0178103, 0.0118092, -0.0187974, 0.0131252, -0.0300570, 0.0306066
6: -0.0014565, 0.0060613, -0.0017905, 0.0070076, -0.0084641, 0.0078517
7: -0.0069059, 0.0125447, -0.0077701, 0.0131929, -0.0200989, 0.0203148
8: -0.0031959, 0.0070330, -0.0036504, 0.0073739, -0.0105698, 0.0106834
9: -0.0100190, 0.0018420, -0.0104142, 0.0023689, -0.0123879, 0.0122562

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135621, upper bound: 0.0125395
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135728, upper bound: 0.0126907
time: 2.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9895927, 1.0107832, 0.9886189, 1.0069369, -0.0173442, 0.0221643
1: -0.0038572, 0.0009682, -0.0040998, 0.0004645, -0.0043217, 0.0050411
2: -0.0161494, 0.0103871, -0.0125158, 0.0116730, -0.0278224, 0.0229029
3: -0.0060009, 0.0056385, -0.0065862, 0.0044235, -0.0104244, 0.0121595
4: -0.0024112, 0.0025383, -0.0018945, 0.0027872, -0.0051706, 0.0044328
5: -0.0201395, 0.0120237, -0.0167822, 0.0136411, -0.0336004, 0.0283916
6: -0.0015109, 0.0085473, -0.0019214, 0.0058003, -0.0073112, 0.0104688
7: -0.0070468, 0.0140743, -0.0081089, 0.0118696, -0.0189164, 0.0220649
8: -0.0032700, 0.0078374, -0.0038285, 0.0066780, -0.0099479, 0.0116037
9: -0.0109517, 0.0019279, -0.0096073, 0.0025755, -0.0134551, 0.0115351

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0132927
time: 2.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135736
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9895927, 1.0107832, 0.9884371, 1.0092145, -0.0196218, 0.0223461
1: -0.0038572, 0.0009682, -0.0041451, 0.0008073, -0.0046645, 0.0051134
2: -0.0161494, 0.0103871, -0.0148090, 0.0119130, -0.0280624, 0.0251961
3: -0.0060009, 0.0056385, -0.0066954, 0.0052504, -0.0112512, 0.0123339
4: -0.0024112, 0.0025383, -0.0022461, 0.0028336, -0.0052448, 0.0047844
5: -0.0201395, 0.0120237, -0.0190669, 0.0139429, -0.0324547, 0.0299507
6: -0.0015109, 0.0085473, -0.0019980, 0.0073169, -0.0088278, 0.0105454
7: -0.0070468, 0.0140743, -0.0083071, 0.0133700, -0.0204167, 0.0223814
8: -0.0032700, 0.0078374, -0.0039328, 0.0074670, -0.0107370, 0.0117702
9: -0.0109517, 0.0019279, -0.0105222, 0.0026964, -0.0136481, 0.0124500

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0133805
time: 2.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0137204
time: 2.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.94 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127015, upper bound: 0.0132045
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0135416
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0137946, upper bound: 0.0131128
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0137946, upper bound: 0.0139577
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0126412, upper bound: 0.0132926
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135728
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0137318, upper bound: 0.0130940
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0137318, upper bound: 0.0140236
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135481, upper bound: 0.0125378
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135576, upper bound: 0.0126875
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135481, upper bound: 0.0126033
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135576, upper bound: 0.0127323
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0132926
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0126949, upper bound: 0.0133805
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0137192
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127651, upper bound: 0.0132045
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0135416
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0138422, upper bound: 0.0131128
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0138422, upper bound: 0.0139577
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127255, upper bound: 0.0132927
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135736
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0137988, upper bound: 0.0130940
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0137988, upper bound: 0.0140246
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135620, upper bound: 0.0125378
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135728, upper bound: 0.0126875
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135621, upper bound: 0.0125395
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0135728, upper bound: 0.0126907
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0132927
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135736
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0133805
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.94
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0137204

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9894493, 1.0069919, -0.0167077, 0.0174196
1: -0.0037574, 0.0006411, -0.0038929, 0.0004782, -0.0041631, 0.0043405
2: -0.0134516, 0.0098585, -0.0125884, 0.0105764, -0.0230024, 0.0220623
3: -0.0057603, 0.0048494, -0.0060870, 0.0044566, -0.0100418, 0.0104697
4: -0.0020756, 0.0024360, -0.0019086, 0.0025749, -0.0044521, 0.0042701
5: -0.0179591, 0.0113588, -0.0168734, 0.0122617, -0.0289310, 0.0277485
6: -0.0013422, 0.0060990, -0.0015713, 0.0058235, -0.0070429, 0.0073430
7: -0.0066102, 0.0126424, -0.0072031, 0.0119295, -0.0182221, 0.0189986
8: -0.0030404, 0.0070844, -0.0033522, 0.0067095, -0.0095828, 0.0099912
9: -0.0100785, 0.0016616, -0.0096438, 0.0020232, -0.0115852, 0.0111117

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0134987
time: 2.20 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0135416
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9899552, 1.0069933, 0.9892929, 1.0096668, -0.0197116, 0.0177004
1: -0.0037669, 0.0004786, -0.0039319, 0.0008537, -0.0045468, 0.0044105
2: -0.0125903, 0.0099084, -0.0151955, 0.0107829, -0.0233732, 0.0251039
3: -0.0057830, 0.0044574, -0.0061810, 0.0053623, -0.0109674, 0.0106384
4: -0.0019089, 0.0024456, -0.0022937, 0.0026149, -0.0045238, 0.0046637
5: -0.0168758, 0.0114216, -0.0193762, 0.0125215, -0.0284659, 0.0303063
6: -0.0013581, 0.0058241, -0.0016373, 0.0076716, -0.0090297, 0.0074614
7: -0.0066514, 0.0119311, -0.0073737, 0.0135730, -0.0199017, 0.0193048
8: -0.0030621, 0.0067103, -0.0034419, 0.0075738, -0.0104661, 0.0101522
9: -0.0096448, 0.0016868, -0.0106460, 0.0021272, -0.0117720, 0.0121360

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0130348
time: 2.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0131128
time: 3.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9898577, 1.0094407, 0.9892929, 1.0096668, -0.0198091, 0.0201477
1: -0.0037912, 0.0008305, -0.0039319, 0.0008537, -0.0046449, 0.0047624
2: -0.0150023, 0.0100372, -0.0151955, 0.0107829, -0.0257852, 0.0252327
3: -0.0058416, 0.0053063, -0.0061810, 0.0053623, -0.0112039, 0.0114873
4: -0.0022699, 0.0024706, -0.0022937, 0.0026149, -0.0048848, 0.0047643
5: -0.0192216, 0.0115836, -0.0193762, 0.0125215, -0.0300707, 0.0289692
6: -0.0013992, 0.0074943, -0.0016373, 0.0076716, -0.0090709, 0.0091315
7: -0.0067578, 0.0134715, -0.0073737, 0.0135730, -0.0203308, 0.0208452
8: -0.0031180, 0.0075204, -0.0034419, 0.0075738, -0.0106918, 0.0109623
9: -0.0105841, 0.0017516, -0.0106460, 0.0021272, -0.0127113, 0.0123976

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0138976
time: 2.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0139577
time: 2.24 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9894493, 1.0069919, -0.0176591, 0.0174037
1: -0.0039844, 0.0005904, -0.0038929, 0.0004782, -0.0044002, 0.0043365
2: -0.0131826, 0.0110613, -0.0125884, 0.0105764, -0.0229814, 0.0233186
3: -0.0063077, 0.0047270, -0.0060870, 0.0044566, -0.0106136, 0.0104601
4: -0.0020236, 0.0026688, -0.0019086, 0.0025749, -0.0044480, 0.0045133
5: -0.0176208, 0.0128716, -0.0168734, 0.0122617, -0.0289045, 0.0293287
6: -0.0017261, 0.0060132, -0.0015713, 0.0058235, -0.0074439, 0.0073363
7: -0.0076036, 0.0124203, -0.0072031, 0.0119295, -0.0192597, 0.0189812
8: -0.0035628, 0.0069676, -0.0033522, 0.0067095, -0.0101285, 0.0099820
9: -0.0099431, 0.0022674, -0.0096438, 0.0020232, -0.0115746, 0.0117445

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135501
time: 2.41 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135728
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890854, 1.0068303, 0.9892929, 1.0096668, -0.0205814, 0.0175374
1: -0.0039836, 0.0004380, -0.0039319, 0.0008537, -0.0047621, 0.0043698
2: -0.0123750, 0.0110568, -0.0151955, 0.0107829, -0.0231579, 0.0262523
3: -0.0063057, 0.0043595, -0.0061810, 0.0053623, -0.0114866, 0.0105405
4: -0.0018673, 0.0026679, -0.0022937, 0.0026149, -0.0044822, 0.0048845
5: -0.0166051, 0.0128660, -0.0193762, 0.0125215, -0.0283872, 0.0317411
6: -0.0017247, 0.0057554, -0.0016373, 0.0076716, -0.0093964, 0.0073926
7: -0.0076000, 0.0117533, -0.0073737, 0.0135730, -0.0208439, 0.0191270
8: -0.0035609, 0.0066168, -0.0034419, 0.0075738, -0.0109616, 0.0100587
9: -0.0095363, 0.0022652, -0.0106460, 0.0021272, -0.0116635, 0.0127105

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0130368
time: 2.29 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0130940
time: 2.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889479, 1.0089519, 0.9892929, 1.0096668, -0.0207189, 0.0196590
1: -0.0040179, 0.0007804, -0.0039319, 0.0008537, -0.0048716, 0.0047122
2: -0.0145846, 0.0112386, -0.0151955, 0.0107829, -0.0253674, 0.0264341
3: -0.0063884, 0.0051854, -0.0061810, 0.0053623, -0.0117507, 0.0113664
4: -0.0022185, 0.0027031, -0.0022937, 0.0026149, -0.0048334, 0.0049968
5: -0.0188873, 0.0130946, -0.0193762, 0.0125215, -0.0298947, 0.0305263
6: -0.0017827, 0.0071108, -0.0016373, 0.0076716, -0.0094544, 0.0087481
7: -0.0077501, 0.0132520, -0.0073737, 0.0135730, -0.0213231, 0.0206257
8: -0.0036398, 0.0074050, -0.0034419, 0.0075738, -0.0112136, 0.0108468
9: -0.0104503, 0.0023567, -0.0106460, 0.0021272, -0.0125774, 0.0130027

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0139888
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0140237
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9903733, 1.0060929, 0.9890482, 1.0075438, -0.0171705, 0.0169462
1: -0.0036627, 0.0002542, -0.0039929, 0.0006158, -0.0042785, 0.0042225
2: -0.0114014, 0.0093564, -0.0133173, 0.0111061, -0.0223772, 0.0226736
3: -0.0055317, 0.0039163, -0.0063281, 0.0047883, -0.0103201, 0.0101851
4: -0.0016788, 0.0023388, -0.0020496, 0.0026775, -0.0043311, 0.0043884
5: -0.0153805, 0.0107273, -0.0177902, 0.0129280, -0.0281447, 0.0285175
6: -0.0011819, 0.0054446, -0.0017404, 0.0060562, -0.0072380, 0.0071434
7: -0.0061955, 0.0109491, -0.0076407, 0.0125315, -0.0187270, 0.0184822
8: -0.0028223, 0.0061939, -0.0035823, 0.0070261, -0.0098484, 0.0097196
9: -0.0090460, 0.0014087, -0.0100109, 0.0022900, -0.0112704, 0.0114196

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0125401
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0125401
time: 2.39 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9899960, 1.0068043, 0.9889880, 1.0079691, -0.0179731, 0.0173110
1: -0.0037567, 0.0004315, -0.0040079, 0.0006796, -0.0044363, 0.0044394
2: -0.0123407, 0.0098545, -0.0137449, 0.0111855, -0.0228590, 0.0235994
3: -0.0057585, 0.0043438, -0.0063643, 0.0049422, -0.0107007, 0.0107081
4: -0.0018606, 0.0024352, -0.0021151, 0.0026928, -0.0045535, 0.0045503
5: -0.0165620, 0.0113538, -0.0182154, 0.0130279, -0.0280775, 0.0295693
6: -0.0013409, 0.0057444, -0.0017658, 0.0063399, -0.0076808, 0.0072972
7: -0.0066069, 0.0117250, -0.0077063, 0.0128108, -0.0194177, 0.0194312
8: -0.0030387, 0.0066019, -0.0036168, 0.0071729, -0.0102116, 0.0102187
9: -0.0095191, 0.0016596, -0.0101812, 0.0023300, -0.0118491, 0.0118408

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0126974
time: 2.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0126974
time: 2.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9903733, 1.0060929, 0.9890209, 1.0079547, -0.0175815, 0.0170720
1: -0.0036627, 0.0002542, -0.0039997, 0.0006781, -0.0043408, 0.0042539
2: -0.0114014, 0.0093564, -0.0137325, 0.0111421, -0.0225435, 0.0230889
3: -0.0055317, 0.0039163, -0.0063445, 0.0049386, -0.0104704, 0.0102608
4: -0.0016788, 0.0023388, -0.0021136, 0.0026844, -0.0043632, 0.0044524
5: -0.0153805, 0.0107273, -0.0182055, 0.0129732, -0.0275524, 0.0289328
6: -0.0011819, 0.0054446, -0.0017519, 0.0063285, -0.0075104, 0.0071965
7: -0.0061955, 0.0109491, -0.0076704, 0.0128043, -0.0189998, 0.0186195
8: -0.0028223, 0.0061939, -0.0035979, 0.0071695, -0.0099918, 0.0097918
9: -0.0090460, 0.0014087, -0.0101772, 0.0023081, -0.0113541, 0.0115860

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0126033
time: 2.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0126032
time: 2.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9899960, 1.0068043, 0.9889568, 1.0085776, -0.0185816, 0.0176009
1: -0.0037567, 0.0004315, -0.0040157, 0.0007420, -0.0044987, 0.0044472
2: -0.0123407, 0.0098545, -0.0142647, 0.0112268, -0.0232418, 0.0241192
3: -0.0057585, 0.0043438, -0.0063831, 0.0050927, -0.0108512, 0.0107269
4: -0.0018606, 0.0024352, -0.0021791, 0.0027008, -0.0045615, 0.0046143
5: -0.0165620, 0.0113538, -0.0186314, 0.0130798, -0.0283491, 0.0299852
6: -0.0013409, 0.0057444, -0.0017790, 0.0068171, -0.0081580, 0.0074194
7: -0.0066069, 0.0117250, -0.0077403, 0.0130839, -0.0196908, 0.0194653
8: -0.0030387, 0.0066019, -0.0036347, 0.0073166, -0.0103552, 0.0102366
9: -0.0095191, 0.0016596, -0.0103478, 0.0023508, -0.0118699, 0.0120074

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0127323
time: 2.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0127323
time: 2.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9898919, 1.0089169, 0.9886201, 1.0068272, -0.0169353, 0.0202968
1: -0.0037826, 0.0007768, -0.0040995, 0.0004372, -0.0042198, 0.0047236
2: -0.0145546, 0.0099919, -0.0123709, 0.0116714, -0.0262259, 0.0223628
3: -0.0058210, 0.0051767, -0.0065854, 0.0043576, -0.0101786, 0.0113937
4: -0.0022148, 0.0024618, -0.0018665, 0.0027869, -0.0048450, 0.0043283
5: -0.0188633, 0.0115266, -0.0165999, 0.0136389, -0.0314841, 0.0276713
6: -0.0013848, 0.0070833, -0.0019209, 0.0057541, -0.0071388, 0.0090041
7: -0.0067204, 0.0132362, -0.0081075, 0.0117499, -0.0184703, 0.0206752
8: -0.0030983, 0.0073967, -0.0038278, 0.0066150, -0.0097134, 0.0108729
9: -0.0104406, 0.0017288, -0.0095343, 0.0025747, -0.0126076, 0.0112631

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135501
time: 2.27 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
time: 2.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9898919, 1.0089169, 0.9884507, 1.0089504, -0.0190585, 0.0204661
1: -0.0037826, 0.0007768, -0.0041417, 0.0007802, -0.0045628, 0.0049185
2: -0.0145546, 0.0099919, -0.0145832, 0.0118950, -0.0264496, 0.0245751
3: -0.0058210, 0.0051767, -0.0066872, 0.0051850, -0.0110060, 0.0118639
4: -0.0022148, 0.0024618, -0.0022183, 0.0028301, -0.0050449, 0.0046801
5: -0.0188633, 0.0115266, -0.0188862, 0.0139202, -0.0306845, 0.0290894
6: -0.0013848, 0.0070833, -0.0019923, 0.0071095, -0.0084943, 0.0090755
7: -0.0067204, 0.0132362, -0.0082922, 0.0132513, -0.0199717, 0.0215285
8: -0.0030983, 0.0073967, -0.0039249, 0.0074046, -0.0105029, 0.0113216
9: -0.0104406, 0.0017288, -0.0104498, 0.0026873, -0.0131280, 0.0121786

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129980, upper bound: 0.0136929
time: 2.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129980, upper bound: 0.0137193
time: 2.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9894315, 1.0069999, -0.0171579, 0.0191207
1: -0.0037951, 0.0007394, -0.0038973, 0.0004802, -0.0042753, 0.0045044
2: -0.0142430, 0.0100580, -0.0125988, 0.0105999, -0.0248429, 0.0226568
3: -0.0058511, 0.0050865, -0.0060977, 0.0044613, -0.0103124, 0.0108650
4: -0.0021764, 0.0024746, -0.0019106, 0.0025795, -0.0046202, 0.0043852
5: -0.0186140, 0.0116097, -0.0168866, 0.0122913, -0.0300233, 0.0276614
6: -0.0014058, 0.0067972, -0.0015788, 0.0058268, -0.0072327, 0.0083761
7: -0.0067750, 0.0130725, -0.0072226, 0.0119381, -0.0187131, 0.0197159
8: -0.0031270, 0.0073106, -0.0033624, 0.0067140, -0.0098410, 0.0103684
9: -0.0103408, 0.0017621, -0.0096491, 0.0020350, -0.0120227, 0.0114112

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134987
time: 2.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134988
time: 2.38 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9897828, 1.0074072, 0.9892764, 1.0096875, -0.0199048, 0.0181307
1: -0.0038098, 0.0005817, -0.0039360, 0.0008559, -0.0046024, 0.0045177
2: -0.0131368, 0.0101360, -0.0152133, 0.0108046, -0.0239414, 0.0253493
3: -0.0058866, 0.0047062, -0.0061909, 0.0053674, -0.0111015, 0.0108971
4: -0.0020147, 0.0024897, -0.0022959, 0.0026191, -0.0046338, 0.0047207
5: -0.0175632, 0.0117079, -0.0193904, 0.0125488, -0.0292345, 0.0306768
6: -0.0014308, 0.0059985, -0.0016442, 0.0076880, -0.0091187, 0.0076427
7: -0.0068394, 0.0123824, -0.0073916, 0.0135824, -0.0201450, 0.0197741
8: -0.0031609, 0.0069477, -0.0034513, 0.0075787, -0.0105941, 0.0103990
9: -0.0099200, 0.0018014, -0.0106517, 0.0021381, -0.0120581, 0.0122843

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0130348
time: 2.41 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0130348
time: 2.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9896832, 1.0103571, 0.9892764, 1.0096875, -0.0200043, 0.0210807
1: -0.0038346, 0.0009245, -0.0039360, 0.0008559, -0.0046905, 0.0048605
2: -0.0157853, 0.0102675, -0.0152133, 0.0108046, -0.0265899, 0.0254808
3: -0.0059465, 0.0055331, -0.0061909, 0.0053674, -0.0113139, 0.0117240
4: -0.0023663, 0.0025152, -0.0022959, 0.0026191, -0.0049854, 0.0048110
5: -0.0198481, 0.0118733, -0.0193904, 0.0125488, -0.0307800, 0.0296471
6: -0.0014727, 0.0082131, -0.0016442, 0.0076880, -0.0091607, 0.0098573
7: -0.0069480, 0.0138829, -0.0073916, 0.0135824, -0.0205304, 0.0212746
8: -0.0032181, 0.0077368, -0.0034513, 0.0075787, -0.0107967, 0.0111881
9: -0.0108350, 0.0018676, -0.0106517, 0.0021381, -0.0129731, 0.0125194

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0138976
time: 2.46 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0138976
time: 2.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9894315, 1.0069999, -0.0179410, 0.0183237
1: -0.0039902, 0.0006576, -0.0038973, 0.0004802, -0.0044704, 0.0044547
2: -0.0135619, 0.0110920, -0.0125988, 0.0105999, -0.0241618, 0.0236908
3: -0.0063217, 0.0048892, -0.0060977, 0.0044613, -0.0107831, 0.0107452
4: -0.0020926, 0.0026747, -0.0019106, 0.0025795, -0.0045692, 0.0045853
5: -0.0180690, 0.0129103, -0.0168866, 0.0122913, -0.0296923, 0.0290949
6: -0.0017359, 0.0061720, -0.0015788, 0.0058268, -0.0075628, 0.0077508
7: -0.0076290, 0.0127146, -0.0072226, 0.0119381, -0.0195672, 0.0194985
8: -0.0035762, 0.0071224, -0.0033624, 0.0067140, -0.0102902, 0.0102541
9: -0.0101226, 0.0022829, -0.0096491, 0.0020350, -0.0118901, 0.0119320

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135501
time: 2.27 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135536
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9890595, 1.0071009, 0.9892764, 1.0096875, -0.0206280, 0.0178244
1: -0.0039900, 0.0005054, -0.0039360, 0.0008559, -0.0048086, 0.0044414
2: -0.0127324, 0.0110912, -0.0152133, 0.0108046, -0.0235370, 0.0263044
3: -0.0063213, 0.0045221, -0.0061909, 0.0053674, -0.0115987, 0.0107130
4: -0.0019364, 0.0026746, -0.0022959, 0.0026191, -0.0045555, 0.0049322
5: -0.0170545, 0.0129092, -0.0193904, 0.0125488, -0.0288198, 0.0320508
6: -0.0017357, 0.0058695, -0.0016442, 0.0076880, -0.0094236, 0.0075136
7: -0.0076283, 0.0120484, -0.0073916, 0.0135824, -0.0210473, 0.0194401
8: -0.0035758, 0.0067720, -0.0034513, 0.0075787, -0.0110686, 0.0102233
9: -0.0097163, 0.0022825, -0.0106517, 0.0021381, -0.0118545, 0.0128345

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0130368
time: 2.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0130393
time: 2.43 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889016, 1.0096251, 0.9892764, 1.0096875, -0.0207860, 0.0203487
1: -0.0040294, 0.0008494, -0.0039360, 0.0008559, -0.0048852, 0.0047854
2: -0.0151598, 0.0112996, -0.0152133, 0.0108046, -0.0259645, 0.0265129
3: -0.0064162, 0.0053520, -0.0061909, 0.0053674, -0.0117836, 0.0115429
4: -0.0022893, 0.0027149, -0.0022959, 0.0026191, -0.0049084, 0.0050108
5: -0.0193477, 0.0131714, -0.0193904, 0.0125488, -0.0303547, 0.0310647
6: -0.0018022, 0.0076389, -0.0016442, 0.0076880, -0.0094902, 0.0092831
7: -0.0078005, 0.0135543, -0.0073916, 0.0135824, -0.0213828, 0.0209459
8: -0.0036663, 0.0075639, -0.0034513, 0.0075787, -0.0112450, 0.0110153
9: -0.0106346, 0.0023874, -0.0106517, 0.0021381, -0.0127727, 0.0130392

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0139888
time: 2.38 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0139933
time: 2.28 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9901279, 1.0064776, 0.9890482, 1.0075438, -0.0174159, 0.0174032
1: -0.0037238, 0.0003501, -0.0039929, 0.0006158, -0.0043396, 0.0043364
2: -0.0119093, 0.0096804, -0.0133173, 0.0111061, -0.0229807, 0.0229976
3: -0.0056792, 0.0041475, -0.0063281, 0.0047883, -0.0104675, 0.0104598
4: -0.0017771, 0.0024015, -0.0020496, 0.0026775, -0.0044479, 0.0044511
5: -0.0160193, 0.0111348, -0.0177902, 0.0129280, -0.0289037, 0.0289250
6: -0.0012853, 0.0056067, -0.0017404, 0.0060562, -0.0073415, 0.0073361
7: -0.0064631, 0.0113686, -0.0076407, 0.0125315, -0.0189946, 0.0189807
8: -0.0029630, 0.0064145, -0.0035823, 0.0070261, -0.0099891, 0.0099817
9: -0.0093018, 0.0015719, -0.0100109, 0.0022900, -0.0115743, 0.0115828

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125401
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125401
time: 2.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9898285, 1.0071468, 0.9889880, 1.0079691, -0.0181407, 0.0177342
1: -0.0037984, 0.0005168, -0.0040079, 0.0006796, -0.0044780, 0.0045247
2: -0.0127930, 0.0100757, -0.0137449, 0.0111855, -0.0234178, 0.0238205
3: -0.0058591, 0.0045497, -0.0063643, 0.0049422, -0.0108013, 0.0109140
4: -0.0019482, 0.0024780, -0.0021151, 0.0026928, -0.0046410, 0.0045931
5: -0.0171307, 0.0116320, -0.0182154, 0.0130279, -0.0287014, 0.0298474
6: -0.0014115, 0.0058888, -0.0017658, 0.0063399, -0.0077514, 0.0074756
7: -0.0067896, 0.0120985, -0.0077063, 0.0128108, -0.0196004, 0.0198047
8: -0.0031347, 0.0067983, -0.0036168, 0.0071729, -0.0103076, 0.0104151
9: -0.0097468, 0.0017710, -0.0101812, 0.0023300, -0.0120768, 0.0119522

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128281, upper bound: 0.0126974
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128280, upper bound: 0.0126973
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9901279, 1.0064776, 0.9890209, 1.0079547, -0.0178269, 0.0174567
1: -0.0037238, 0.0003501, -0.0039997, 0.0006781, -0.0044019, 0.0043497
2: -0.0119093, 0.0096804, -0.0137325, 0.0111421, -0.0230514, 0.0234128
3: -0.0056792, 0.0041475, -0.0063445, 0.0049386, -0.0106178, 0.0104920
4: -0.0017771, 0.0024015, -0.0021136, 0.0026844, -0.0044615, 0.0045151
5: -0.0160193, 0.0111348, -0.0182055, 0.0129732, -0.0281256, 0.0293403
6: -0.0012853, 0.0056067, -0.0017519, 0.0063285, -0.0076138, 0.0073586
7: -0.0064631, 0.0113686, -0.0076704, 0.0128043, -0.0192674, 0.0190390
8: -0.0029630, 0.0064145, -0.0035979, 0.0071695, -0.0101325, 0.0100124
9: -0.0093018, 0.0015719, -0.0101772, 0.0023081, -0.0116099, 0.0117491

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125395
time: 2.22 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125395
time: 2.38 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9898285, 1.0071468, 0.9889568, 1.0085776, -0.0187491, 0.0180234
1: -0.0037984, 0.0005168, -0.0040157, 0.0007420, -0.0045404, 0.0045325
2: -0.0127930, 0.0100757, -0.0142647, 0.0112268, -0.0237997, 0.0243403
3: -0.0058591, 0.0045497, -0.0063831, 0.0050927, -0.0109519, 0.0109327
4: -0.0019482, 0.0024780, -0.0021791, 0.0027008, -0.0046490, 0.0046571
5: -0.0171307, 0.0116320, -0.0186314, 0.0130798, -0.0288098, 0.0302634
6: -0.0014115, 0.0058888, -0.0017790, 0.0068171, -0.0082286, 0.0075975
7: -0.0067896, 0.0120985, -0.0077403, 0.0130839, -0.0198735, 0.0198388
8: -0.0031347, 0.0067983, -0.0036347, 0.0073166, -0.0104513, 0.0104330
9: -0.0097468, 0.0017710, -0.0103478, 0.0023508, -0.0120976, 0.0121188

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128281, upper bound: 0.0126907
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128281, upper bound: 0.0126907
time: 2.48 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0096927, 0.9886416, 1.0068365, -0.0171147, 0.0210511
1: -0.0038250, 0.0008564, -0.0040942, 0.0004395, -0.0042645, 0.0048363
2: -0.0152176, 0.0102166, -0.0123832, 0.0116429, -0.0268605, 0.0225998
3: -0.0059233, 0.0053687, -0.0065725, 0.0043632, -0.0102864, 0.0116656
4: -0.0022964, 0.0025053, -0.0018689, 0.0027813, -0.0049606, 0.0043741
5: -0.0193939, 0.0118092, -0.0166154, 0.0136032, -0.0322357, 0.0280178
6: -0.0014565, 0.0076919, -0.0019118, 0.0057580, -0.0072145, 0.0096037
7: -0.0069059, 0.0135846, -0.0080840, 0.0117600, -0.0186660, 0.0211687
8: -0.0031959, 0.0075799, -0.0038155, 0.0066204, -0.0098163, 0.0111324
9: -0.0106531, 0.0018420, -0.0095405, 0.0025603, -0.0129086, 0.0113824

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135501
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135536
time: 2.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0096927, 0.9884640, 1.0089737, -0.0192519, 0.0212287
1: -0.0038250, 0.0008564, -0.0041384, 0.0007826, -0.0046076, 0.0049948
2: -0.0152176, 0.0102166, -0.0146033, 0.0118775, -0.0270951, 0.0248198
3: -0.0059233, 0.0053687, -0.0066793, 0.0051908, -0.0111140, 0.0120479
4: -0.0022964, 0.0025053, -0.0022208, 0.0028268, -0.0051232, 0.0047261
5: -0.0193939, 0.0118092, -0.0189023, 0.0138983, -0.0312506, 0.0295759
6: -0.0014565, 0.0076919, -0.0019867, 0.0071280, -0.0085844, 0.0096786
7: -0.0069059, 0.0135846, -0.0082778, 0.0132618, -0.0201678, 0.0218624
8: -0.0031959, 0.0075799, -0.0039174, 0.0074101, -0.0106060, 0.0114972
9: -0.0106531, 0.0018420, -0.0104563, 0.0026785, -0.0133316, 0.0122982

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130257, upper bound: 0.0136929
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130257, upper bound: 0.0136979
time: 2.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.58 seconds
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0134987
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0135416
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0130348
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0131128
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0138976
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130367, upper bound: 0.0139577
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135501
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135728
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0130368
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0130940
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0139888
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0129758, upper bound: 0.0140237
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0125401
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0125401
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0126974
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0126974
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0126033
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127881, upper bound: 0.0126032
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0127323
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127884, upper bound: 0.0127323
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135501
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0129980, upper bound: 0.0136929
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0129980, upper bound: 0.0137193
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134987
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134988
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0130348
NS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0130348
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0138976
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0131128, upper bound: 0.0138976
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135501
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135536
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0130368
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0130393
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0139888
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130763, upper bound: 0.0139933
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125401
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125401
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128281, upper bound: 0.0126974
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128280, upper bound: 0.0126973
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125395
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128269, upper bound: 0.0125395
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128281, upper bound: 0.0126907
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0128281, upper bound: 0.0126907
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135501
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135536
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130257, upper bound: 0.0136929
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.58
Output dim: 0, lower bound: -0.0130257, upper bound: 0.0136979

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9899784, 1.0069005, -0.0166298, 0.0168194
1: -0.0037574, 0.0006411, -0.0037611, 0.0004555, -0.0041437, 0.0041909
2: -0.0134516, 0.0098585, -0.0124677, 0.0098777, -0.0222098, 0.0219594
3: -0.0057603, 0.0048494, -0.0057690, 0.0044017, -0.0099950, 0.0101089
4: -0.0020756, 0.0024360, -0.0018852, 0.0024397, -0.0042987, 0.0042502
5: -0.0179591, 0.0113588, -0.0167217, 0.0113830, -0.0279341, 0.0276192
6: -0.0013422, 0.0060990, -0.0013483, 0.0057850, -0.0070100, 0.0070900
7: -0.0066102, 0.0126424, -0.0066261, 0.0118299, -0.0181371, 0.0183439
8: -0.0030404, 0.0070844, -0.0030487, 0.0066571, -0.0095381, 0.0096469
9: -0.0100785, 0.0016616, -0.0095830, 0.0016713, -0.0111860, 0.0110599

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0128255
time: 2.18 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0134988
time: 2.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9898068, 1.0072820, -0.0171006, 0.0172117
1: -0.0037574, 0.0006411, -0.0038038, 0.0005505, -0.0042610, 0.0042887
2: -0.0134516, 0.0098585, -0.0129715, 0.0101043, -0.0227278, 0.0225811
3: -0.0057603, 0.0048494, -0.0058722, 0.0046309, -0.0102779, 0.0103447
4: -0.0020756, 0.0024360, -0.0019827, 0.0024836, -0.0043989, 0.0043705
5: -0.0179591, 0.0113588, -0.0173553, 0.0116680, -0.0285856, 0.0284011
6: -0.0013422, 0.0060990, -0.0014206, 0.0059458, -0.0072085, 0.0072553
7: -0.0066102, 0.0126424, -0.0068132, 0.0122459, -0.0186506, 0.0187718
8: -0.0030404, 0.0070844, -0.0031472, 0.0068759, -0.0098082, 0.0098719
9: -0.0100785, 0.0016616, -0.0098368, 0.0017854, -0.0114469, 0.0113730

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0128858
time: 2.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0135416
time: 2.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9898577, 1.0094407, 0.9898577, 1.0094407, -0.0195830, 0.0195830
1: -0.0037912, 0.0008305, -0.0037912, 0.0008305, -0.0046217, 0.0046217
2: -0.0150023, 0.0100372, -0.0150023, 0.0100372, -0.0250395, 0.0250395
3: -0.0058416, 0.0053063, -0.0058416, 0.0053063, -0.0111479, 0.0111479
4: -0.0022699, 0.0024706, -0.0022699, 0.0024706, -0.0047405, 0.0047405
5: -0.0192216, 0.0115836, -0.0192216, 0.0115836, -0.0288262, 0.0288262
6: -0.0013992, 0.0074943, -0.0013992, 0.0074943, -0.0088935, 0.0088935
7: -0.0067578, 0.0134715, -0.0067578, 0.0134715, -0.0202293, 0.0202293
8: -0.0031180, 0.0075204, -0.0031180, 0.0075204, -0.0106384, 0.0106384
9: -0.0105841, 0.0017516, -0.0105841, 0.0017516, -0.0123357, 0.0123357

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128934, upper bound: 0.0131975
time: 2.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129609, upper bound: 0.0136196
time: 2.11 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898577, 1.0094407, 0.9896832, 1.0102810, -0.0204233, 0.0197574
1: -0.0037912, 0.0008305, -0.0038346, 0.0009167, -0.0047079, 0.0046652
2: -0.0150023, 0.0100372, -0.0157202, 0.0102675, -0.0252698, 0.0257574
3: -0.0058416, 0.0053063, -0.0059465, 0.0055142, -0.0113558, 0.0112528
4: -0.0022699, 0.0024706, -0.0023583, 0.0025151, -0.0047851, 0.0048289
5: -0.0192216, 0.0115836, -0.0197960, 0.0118733, -0.0295008, 0.0294789
6: -0.0013992, 0.0074943, -0.0014727, 0.0081534, -0.0095526, 0.0089670
7: -0.0067578, 0.0134715, -0.0069480, 0.0138488, -0.0206066, 0.0204195
8: -0.0031180, 0.0075204, -0.0032180, 0.0077188, -0.0108368, 0.0107384
9: -0.0105841, 0.0017516, -0.0108142, 0.0018676, -0.0124518, 0.0125658

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128934, upper bound: 0.0132777
time: 2.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129609, upper bound: 0.0136723
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9899784, 1.0069005, -0.0175795, 0.0168034
1: -0.0039844, 0.0005904, -0.0037611, 0.0004555, -0.0043803, 0.0041870
2: -0.0131826, 0.0110613, -0.0124677, 0.0098777, -0.0221887, 0.0232136
3: -0.0063077, 0.0047270, -0.0057690, 0.0044017, -0.0105658, 0.0100993
4: -0.0020236, 0.0026688, -0.0018852, 0.0024397, -0.0042946, 0.0044929
5: -0.0176208, 0.0128716, -0.0167217, 0.0113830, -0.0279076, 0.0291966
6: -0.0017261, 0.0060132, -0.0013483, 0.0057850, -0.0074104, 0.0070832
7: -0.0076036, 0.0124203, -0.0066261, 0.0118299, -0.0191730, 0.0183265
8: -0.0035628, 0.0069676, -0.0030487, 0.0066571, -0.0100829, 0.0096377
9: -0.0099431, 0.0022674, -0.0095830, 0.0016713, -0.0111754, 0.0116916

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0127817
time: 2.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135501
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9898068, 1.0072820, -0.0180503, 0.0171957
1: -0.0039844, 0.0005904, -0.0038038, 0.0005505, -0.0044977, 0.0042847
2: -0.0131826, 0.0110613, -0.0129715, 0.0101043, -0.0227068, 0.0238353
3: -0.0063077, 0.0047270, -0.0058722, 0.0046309, -0.0108488, 0.0103351
4: -0.0020236, 0.0026688, -0.0019827, 0.0024836, -0.0043948, 0.0046133
5: -0.0176208, 0.0128716, -0.0173553, 0.0116680, -0.0285591, 0.0299785
6: -0.0017261, 0.0060132, -0.0014206, 0.0059458, -0.0076089, 0.0072486
7: -0.0076036, 0.0124203, -0.0068132, 0.0122459, -0.0196864, 0.0187544
8: -0.0035628, 0.0069676, -0.0031472, 0.0068759, -0.0103529, 0.0098627
9: -0.0099431, 0.0022674, -0.0098368, 0.0017854, -0.0114363, 0.0120047

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0128281
time: 2.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135728
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889479, 1.0089519, 0.9898577, 1.0094407, -0.0204928, 0.0190942
1: -0.0040179, 0.0007804, -0.0037912, 0.0008305, -0.0048484, 0.0045715
2: -0.0145846, 0.0112386, -0.0150023, 0.0100372, -0.0246217, 0.0262409
3: -0.0063884, 0.0051854, -0.0058416, 0.0053063, -0.0116948, 0.0110270
4: -0.0022185, 0.0027031, -0.0022699, 0.0024706, -0.0046890, 0.0049730
5: -0.0188873, 0.0130946, -0.0192216, 0.0115836, -0.0286485, 0.0303844
6: -0.0017827, 0.0071108, -0.0013992, 0.0074943, -0.0092770, 0.0085100
7: -0.0077501, 0.0132520, -0.0067578, 0.0134715, -0.0212216, 0.0200098
8: -0.0036398, 0.0074050, -0.0031180, 0.0075204, -0.0111602, 0.0105230
9: -0.0104503, 0.0023567, -0.0105841, 0.0017516, -0.0122019, 0.0129408

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128593, upper bound: 0.0133449
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129197, upper bound: 0.0136928
time: 2.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9889479, 1.0089519, 0.9896832, 1.0102810, -0.0213331, 0.0192687
1: -0.0040179, 0.0007804, -0.0038346, 0.0009167, -0.0049346, 0.0046150
2: -0.0145846, 0.0112386, -0.0157202, 0.0102675, -0.0248521, 0.0269588
3: -0.0063884, 0.0051854, -0.0059465, 0.0055142, -0.0119026, 0.0111318
4: -0.0022185, 0.0027031, -0.0023583, 0.0025151, -0.0047336, 0.0050614
5: -0.0188873, 0.0130946, -0.0197960, 0.0118733, -0.0293247, 0.0310359
6: -0.0017827, 0.0071108, -0.0014727, 0.0081534, -0.0099361, 0.0085835
7: -0.0077501, 0.0132520, -0.0069480, 0.0138488, -0.0215988, 0.0202000
8: -0.0036398, 0.0074050, -0.0032180, 0.0077188, -0.0113586, 0.0106230
9: -0.0104503, 0.0023567, -0.0108142, 0.0018676, -0.0123179, 0.0131709

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128593, upper bound: 0.0133805
time: 2.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129197, upper bound: 0.0137193
time: 2.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9898919, 1.0089169, 0.9891081, 1.0067297, -0.0168378, 0.0198088
1: -0.0037826, 0.0007768, -0.0039779, 0.0004129, -0.0041955, 0.0045958
2: -0.0145546, 0.0099919, -0.0122422, 0.0110269, -0.0255815, 0.0222341
3: -0.0058210, 0.0051767, -0.0062921, 0.0042990, -0.0101200, 0.0110855
4: -0.0022148, 0.0024618, -0.0018416, 0.0026621, -0.0047139, 0.0043034
5: -0.0188633, 0.0115266, -0.0164380, 0.0128284, -0.0306326, 0.0275157
6: -0.0013848, 0.0070833, -0.0017152, 0.0057130, -0.0070977, 0.0087984
7: -0.0067204, 0.0132362, -0.0075753, 0.0116436, -0.0183640, 0.0201160
8: -0.0030983, 0.0073967, -0.0035479, 0.0065591, -0.0096575, 0.0105788
9: -0.0104406, 0.0017288, -0.0094695, 0.0022501, -0.0122667, 0.0111983

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0134333
time: 2.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135501
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898919, 1.0089169, 0.9890830, 1.0069813, -0.0170893, 0.0198339
1: -0.0037826, 0.0007768, -0.0039842, 0.0004756, -0.0042582, 0.0046390
2: -0.0145546, 0.0099919, -0.0125743, 0.0110602, -0.0256148, 0.0225663
3: -0.0058210, 0.0051767, -0.0063072, 0.0044502, -0.0102712, 0.0111898
4: -0.0022148, 0.0024618, -0.0019058, 0.0026686, -0.0047583, 0.0043676
5: -0.0188633, 0.0115266, -0.0168557, 0.0128703, -0.0309209, 0.0281272
6: -0.0013848, 0.0070833, -0.0017258, 0.0058190, -0.0072038, 0.0088090
7: -0.0067204, 0.0132362, -0.0076027, 0.0119179, -0.0186383, 0.0203053
8: -0.0030983, 0.0073967, -0.0035624, 0.0067034, -0.0098017, 0.0106784
9: -0.0104406, 0.0017288, -0.0096367, 0.0022669, -0.0123821, 0.0113656

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0134942
time: 2.37 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135728
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9898919, 1.0089169, 0.9889733, 1.0087116, -0.0188197, 0.0199436
1: -0.0037826, 0.0007768, -0.0040115, 0.0007557, -0.0045383, 0.0047883
2: -0.0145546, 0.0099919, -0.0143792, 0.0112051, -0.0257596, 0.0243711
3: -0.0058210, 0.0051767, -0.0063732, 0.0051259, -0.0109469, 0.0115499
4: -0.0022148, 0.0024618, -0.0021932, 0.0026966, -0.0049114, 0.0046550
5: -0.0188633, 0.0115266, -0.0187230, 0.0130525, -0.0295178, 0.0289344
6: -0.0013848, 0.0070833, -0.0017720, 0.0069223, -0.0083070, 0.0088553
7: -0.0067204, 0.0132362, -0.0077224, 0.0131441, -0.0198645, 0.0209586
8: -0.0030983, 0.0073967, -0.0036253, 0.0073482, -0.0104466, 0.0110219
9: -0.0104406, 0.0017288, -0.0103845, 0.0023398, -0.0127805, 0.0121133

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0135759
time: 2.40 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136929
time: 2.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898919, 1.0089169, 0.9889344, 1.0093356, -0.0194438, 0.0199825
1: -0.0037826, 0.0007768, -0.0040212, 0.0008197, -0.0046024, 0.0047980
2: -0.0145546, 0.0099919, -0.0149125, 0.0112562, -0.0258108, 0.0249045
3: -0.0058210, 0.0051767, -0.0063965, 0.0052803, -0.0111014, 0.0115731
4: -0.0022148, 0.0024618, -0.0022589, 0.0027065, -0.0049213, 0.0047207
5: -0.0188633, 0.0115266, -0.0191498, 0.0131168, -0.0299408, 0.0295431
6: -0.0013848, 0.0070833, -0.0017884, 0.0074119, -0.0087966, 0.0088716
7: -0.0067204, 0.0132362, -0.0077646, 0.0134243, -0.0201448, 0.0210009
8: -0.0030983, 0.0073967, -0.0036475, 0.0074956, -0.0105939, 0.0110442
9: -0.0104406, 0.0017288, -0.0105554, 0.0023656, -0.0128062, 0.0122842

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136368
time: 2.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0137192
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9899784, 1.0069005, -0.0170586, 0.0185738
1: -0.0037951, 0.0007394, -0.0037611, 0.0004555, -0.0042505, 0.0043322
2: -0.0142430, 0.0100580, -0.0124677, 0.0098777, -0.0241207, 0.0225257
3: -0.0058511, 0.0050865, -0.0057690, 0.0044017, -0.0102527, 0.0104497
4: -0.0021764, 0.0024746, -0.0018852, 0.0024397, -0.0044436, 0.0043598
5: -0.0186140, 0.0116097, -0.0167217, 0.0113830, -0.0288757, 0.0274364
6: -0.0014058, 0.0067972, -0.0013483, 0.0057850, -0.0071908, 0.0081455
7: -0.0067750, 0.0130725, -0.0066261, 0.0118299, -0.0186048, 0.0189623
8: -0.0031270, 0.0073106, -0.0030487, 0.0066571, -0.0097841, 0.0099721
9: -0.0103408, 0.0017621, -0.0095830, 0.0016713, -0.0115631, 0.0113451

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0128254
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134988
time: 2.26 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9898068, 1.0072820, -0.0173685, 0.0187454
1: -0.0037951, 0.0007394, -0.0038038, 0.0005505, -0.0043456, 0.0042942
2: -0.0142430, 0.0100580, -0.0129715, 0.0101043, -0.0243473, 0.0229349
3: -0.0058511, 0.0050865, -0.0058722, 0.0046309, -0.0104820, 0.0103581
4: -0.0021764, 0.0024746, -0.0019827, 0.0024836, -0.0044046, 0.0044573
5: -0.0186140, 0.0116097, -0.0173553, 0.0116680, -0.0286225, 0.0278518
6: -0.0014058, 0.0067972, -0.0014206, 0.0059458, -0.0073214, 0.0082179
7: -0.0067750, 0.0130725, -0.0068132, 0.0122459, -0.0190209, 0.0187960
8: -0.0031270, 0.0073106, -0.0031472, 0.0068759, -0.0100029, 0.0098846
9: -0.0103408, 0.0017621, -0.0098368, 0.0017854, -0.0114617, 0.0115989

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0128255
time: 2.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134987
time: 2.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9896832, 1.0103571, 0.9898577, 1.0094407, -0.0197574, 0.0204995
1: -0.0038346, 0.0009245, -0.0037912, 0.0008305, -0.0046652, 0.0047157
2: -0.0157853, 0.0102675, -0.0150023, 0.0100372, -0.0258225, 0.0252698
3: -0.0059465, 0.0055331, -0.0058416, 0.0053063, -0.0112528, 0.0113747
4: -0.0023663, 0.0025152, -0.0022699, 0.0024706, -0.0048369, 0.0047851
5: -0.0198481, 0.0118733, -0.0192216, 0.0115836, -0.0295284, 0.0295012
6: -0.0014727, 0.0082131, -0.0013992, 0.0074943, -0.0089670, 0.0096123
7: -0.0069480, 0.0138829, -0.0067578, 0.0134715, -0.0204195, 0.0206407
8: -0.0032181, 0.0077368, -0.0031180, 0.0075204, -0.0107384, 0.0108547
9: -0.0108350, 0.0018676, -0.0105841, 0.0017516, -0.0125866, 0.0124517

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129526, upper bound: 0.0131975
time: 2.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130266, upper bound: 0.0136196
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9896832, 1.0103571, 0.9896832, 1.0102810, -0.0205978, 0.0206739
1: -0.0038346, 0.0009245, -0.0038346, 0.0009167, -0.0047513, 0.0047591
2: -0.0157853, 0.0102675, -0.0157202, 0.0102675, -0.0260528, 0.0259878
3: -0.0059465, 0.0055331, -0.0059465, 0.0055142, -0.0114607, 0.0114795
4: -0.0023663, 0.0025152, -0.0023583, 0.0025151, -0.0048815, 0.0048735
5: -0.0198481, 0.0118733, -0.0197960, 0.0118733, -0.0298522, 0.0298114
6: -0.0014727, 0.0082131, -0.0014727, 0.0081534, -0.0096261, 0.0096858
7: -0.0069480, 0.0138829, -0.0069480, 0.0138488, -0.0207968, 0.0208310
8: -0.0032181, 0.0077368, -0.0032180, 0.0077188, -0.0109368, 0.0109548
9: -0.0108350, 0.0018676, -0.0108142, 0.0018676, -0.0127026, 0.0126818

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129525, upper bound: 0.0131975
time: 2.39 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130266, upper bound: 0.0136196
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9899784, 1.0069005, -0.0178417, 0.0177768
1: -0.0039902, 0.0006576, -0.0037611, 0.0004555, -0.0044457, 0.0042825
2: -0.0135619, 0.0110920, -0.0124677, 0.0098777, -0.0234396, 0.0235598
3: -0.0063217, 0.0048892, -0.0057690, 0.0044017, -0.0107234, 0.0103299
4: -0.0020926, 0.0026747, -0.0018852, 0.0024397, -0.0043926, 0.0045599
5: -0.0180690, 0.0129103, -0.0167217, 0.0113830, -0.0285447, 0.0287445
6: -0.0017359, 0.0061720, -0.0013483, 0.0057850, -0.0075209, 0.0075203
7: -0.0076290, 0.0127146, -0.0066261, 0.0118299, -0.0194589, 0.0187449
8: -0.0035762, 0.0071224, -0.0030487, 0.0066571, -0.0102332, 0.0098577
9: -0.0101226, 0.0022829, -0.0095830, 0.0016713, -0.0114305, 0.0118659

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0127817
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135501
time: 2.35 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9898068, 1.0072820, -0.0182232, 0.0179484
1: -0.0039902, 0.0006576, -0.0038038, 0.0005505, -0.0045407, 0.0042834
2: -0.0135619, 0.0110920, -0.0129715, 0.0101043, -0.0236662, 0.0240635
3: -0.0063217, 0.0048892, -0.0058722, 0.0046309, -0.0109527, 0.0103320
4: -0.0020926, 0.0026747, -0.0019827, 0.0024836, -0.0043935, 0.0046574
5: -0.0180690, 0.0129103, -0.0173553, 0.0116680, -0.0285506, 0.0292924
6: -0.0017359, 0.0061720, -0.0014206, 0.0059458, -0.0076817, 0.0075926
7: -0.0076290, 0.0127146, -0.0068132, 0.0122459, -0.0198749, 0.0187487
8: -0.0035762, 0.0071224, -0.0031472, 0.0068759, -0.0104520, 0.0098598
9: -0.0101226, 0.0022829, -0.0098368, 0.0017854, -0.0114329, 0.0121197

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0127849
time: 2.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135536
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889016, 1.0096251, 0.9898577, 1.0094407, -0.0205391, 0.0197674
1: -0.0040294, 0.0008494, -0.0037912, 0.0008305, -0.0048599, 0.0046406
2: -0.0151598, 0.0112996, -0.0150023, 0.0100372, -0.0251970, 0.0263019
3: -0.0064162, 0.0053520, -0.0058416, 0.0053063, -0.0117225, 0.0111936
4: -0.0022893, 0.0027149, -0.0022699, 0.0024706, -0.0047599, 0.0049848
5: -0.0193477, 0.0131714, -0.0192216, 0.0115836, -0.0290910, 0.0307517
6: -0.0018022, 0.0076389, -0.0013992, 0.0074943, -0.0092965, 0.0090381
7: -0.0078005, 0.0135543, -0.0067578, 0.0134715, -0.0212720, 0.0203121
8: -0.0036663, 0.0075639, -0.0031180, 0.0075204, -0.0111867, 0.0106819
9: -0.0106346, 0.0023874, -0.0105841, 0.0017516, -0.0123862, 0.0129716

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129285, upper bound: 0.0133449
time: 2.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129988, upper bound: 0.0136929
time: 2.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9889016, 1.0096251, 0.9896832, 1.0102810, -0.0213794, 0.0199419
1: -0.0040294, 0.0008494, -0.0038346, 0.0009167, -0.0049461, 0.0046841
2: -0.0151598, 0.0112996, -0.0157202, 0.0102675, -0.0254274, 0.0270198
3: -0.0064162, 0.0053520, -0.0059465, 0.0055142, -0.0119304, 0.0112984
4: -0.0022893, 0.0027149, -0.0023583, 0.0025151, -0.0048045, 0.0050732
5: -0.0193477, 0.0131714, -0.0197960, 0.0118733, -0.0295694, 0.0312290
6: -0.0018022, 0.0076389, -0.0014727, 0.0081534, -0.0099556, 0.0091117
7: -0.0078005, 0.0135543, -0.0069480, 0.0138488, -0.0216492, 0.0205023
8: -0.0036663, 0.0075639, -0.0032180, 0.0077188, -0.0113851, 0.0107820
9: -0.0106346, 0.0023874, -0.0108142, 0.0018676, -0.0125022, 0.0132016

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0129285, upper bound: 0.0133461
time: 2.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129988, upper bound: 0.0136979
time: 2.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0096927, 0.9891081, 1.0067297, -0.0170079, 0.0205846
1: -0.0038250, 0.0008564, -0.0039779, 0.0004129, -0.0042379, 0.0047031
2: -0.0152176, 0.0102166, -0.0122422, 0.0110269, -0.0262445, 0.0224588
3: -0.0059233, 0.0053687, -0.0062921, 0.0042990, -0.0102223, 0.0113444
4: -0.0022964, 0.0025053, -0.0018416, 0.0026621, -0.0048240, 0.0043468
5: -0.0193939, 0.0118092, -0.0164380, 0.0128284, -0.0313481, 0.0278984
6: -0.0014565, 0.0076919, -0.0017152, 0.0057130, -0.0071694, 0.0094071
7: -0.0069059, 0.0135846, -0.0075753, 0.0116436, -0.0185495, 0.0205858
8: -0.0031959, 0.0075799, -0.0035479, 0.0065591, -0.0097550, 0.0108259
9: -0.0106531, 0.0018420, -0.0094695, 0.0022501, -0.0125531, 0.0113114

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134333
time: 2.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135501
time: 2.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0096927, 0.9890830, 1.0069813, -0.0172594, 0.0206097
1: -0.0038250, 0.0008564, -0.0039842, 0.0004756, -0.0043006, 0.0046855
2: -0.0152176, 0.0102166, -0.0125743, 0.0110602, -0.0262778, 0.0227909
3: -0.0059233, 0.0053687, -0.0063072, 0.0044502, -0.0103734, 0.0113018
4: -0.0022964, 0.0025053, -0.0019058, 0.0026686, -0.0048059, 0.0044111
5: -0.0193939, 0.0118092, -0.0168557, 0.0128703, -0.0312304, 0.0281753
6: -0.0014565, 0.0076919, -0.0017258, 0.0058190, -0.0072755, 0.0094177
7: -0.0069059, 0.0135846, -0.0076027, 0.0119179, -0.0188238, 0.0205086
8: -0.0031959, 0.0075799, -0.0035624, 0.0067034, -0.0098993, 0.0107853
9: -0.0106531, 0.0018420, -0.0096367, 0.0022669, -0.0125060, 0.0114787

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134362
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135536
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0096927, 0.9889733, 1.0087116, -0.0189897, 0.0207194
1: -0.0038250, 0.0008564, -0.0040115, 0.0007557, -0.0045807, 0.0048679
2: -0.0152176, 0.0102166, -0.0143792, 0.0112051, -0.0264227, 0.0245957
3: -0.0059233, 0.0053687, -0.0063732, 0.0051259, -0.0110492, 0.0117419
4: -0.0022964, 0.0025053, -0.0021932, 0.0026966, -0.0049930, 0.0046985
5: -0.0193939, 0.0118092, -0.0187230, 0.0130525, -0.0301221, 0.0295146
6: -0.0014565, 0.0076919, -0.0017720, 0.0069223, -0.0083787, 0.0094640
7: -0.0069059, 0.0135846, -0.0077224, 0.0131441, -0.0200500, 0.0213070
8: -0.0031959, 0.0075799, -0.0036253, 0.0073482, -0.0105441, 0.0112052
9: -0.0106531, 0.0018420, -0.0103845, 0.0023398, -0.0129929, 0.0122264

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129763, upper bound: 0.0135759
time: 2.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129763, upper bound: 0.0136928
time: 2.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9897218, 1.0096927, 0.9889344, 1.0093356, -0.0196138, 0.0207583
1: -0.0038250, 0.0008564, -0.0040212, 0.0008197, -0.0046448, 0.0048776
2: -0.0152176, 0.0102166, -0.0149125, 0.0112562, -0.0264738, 0.0251291
3: -0.0059233, 0.0053687, -0.0063965, 0.0052803, -0.0112036, 0.0117651
4: -0.0022964, 0.0025053, -0.0022589, 0.0027065, -0.0050029, 0.0047641
5: -0.0193939, 0.0118092, -0.0191498, 0.0131168, -0.0303374, 0.0297298
6: -0.0014565, 0.0076919, -0.0017884, 0.0074119, -0.0088684, 0.0094803
7: -0.0069059, 0.0135846, -0.0077646, 0.0134243, -0.0203303, 0.0213493
8: -0.0031959, 0.0075799, -0.0036475, 0.0074956, -0.0106915, 0.0112274
9: -0.0106531, 0.0018420, -0.0105554, 0.0023656, -0.0130187, 0.0123973

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0135781
time: 2.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136978
time: 2.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 8.40 seconds
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0128255
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0134988
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0128858
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0135416
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0128934, upper bound: 0.0131975
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129609, upper bound: 0.0136196
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0128934, upper bound: 0.0132777
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129609, upper bound: 0.0136723
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0127817
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135501
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0128281
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126450, upper bound: 0.0135728
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0128593, upper bound: 0.0133449
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129197, upper bound: 0.0136928
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0128593, upper bound: 0.0133805
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129197, upper bound: 0.0137193
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0134333
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135501
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0134942
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135728
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0135759
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136929
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136368
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0137192
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0128254
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134988
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0128255
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0134987
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129526, upper bound: 0.0131975
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0130266, upper bound: 0.0136196
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129525, upper bound: 0.0131975
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0130266, upper bound: 0.0136196
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0127817
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135501
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0127849
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127340, upper bound: 0.0135536
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129285, upper bound: 0.0133449
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129988, upper bound: 0.0136929
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129285, upper bound: 0.0133461
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129988, upper bound: 0.0136979
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134333
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135501
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134362
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135536
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129763, upper bound: 0.0135759
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129763, upper bound: 0.0136928
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0135781
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.40
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136978

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9899774, 1.0079833, 0.9899784, 1.0069005, -0.0166457, 0.0171847
1: -0.0037613, 0.0007253, -0.0037611, 0.0004555, -0.0041477, 0.0042820
2: -0.0138975, 0.0098791, -0.0124677, 0.0098777, -0.0226923, 0.0219805
3: -0.0057696, 0.0050524, -0.0057690, 0.0044017, -0.0100046, 0.0103285
4: -0.0021620, 0.0024400, -0.0018852, 0.0024397, -0.0043920, 0.0042543
5: -0.0185200, 0.0113847, -0.0167217, 0.0113830, -0.0285409, 0.0276457
6: -0.0013487, 0.0062414, -0.0013483, 0.0057850, -0.0070168, 0.0072440
7: -0.0066272, 0.0130108, -0.0066261, 0.0118299, -0.0181545, 0.0187424
8: -0.0030493, 0.0072781, -0.0030487, 0.0066571, -0.0095473, 0.0098564
9: -0.0103032, 0.0016720, -0.0095830, 0.0016713, -0.0114290, 0.0110705

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125019, upper bound: 0.0134850
time: 2.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125019, upper bound: 0.0135019
time: 2.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9899774, 1.0079833, 0.9898068, 1.0072820, -0.0171165, 0.0175771
1: -0.0037613, 0.0007253, -0.0038038, 0.0005505, -0.0042650, 0.0043797
2: -0.0138975, 0.0098791, -0.0129715, 0.0101043, -0.0232103, 0.0226022
3: -0.0057696, 0.0050524, -0.0058722, 0.0046309, -0.0102875, 0.0105643
4: -0.0021620, 0.0024400, -0.0019827, 0.0024836, -0.0044923, 0.0043746
5: -0.0185200, 0.0113847, -0.0173553, 0.0116680, -0.0291925, 0.0284276
6: -0.0013487, 0.0062414, -0.0014206, 0.0059458, -0.0072152, 0.0074094
7: -0.0066272, 0.0130108, -0.0068132, 0.0122459, -0.0186680, 0.0191703
8: -0.0030493, 0.0072781, -0.0031472, 0.0068759, -0.0098173, 0.0100815
9: -0.0103032, 0.0016720, -0.0098368, 0.0017854, -0.0116899, 0.0113837

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125019, upper bound: 0.0135253
time: 2.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125019, upper bound: 0.0135416
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9899768, 1.0084147, 0.9898834, 1.0092132, -0.0192364, 0.0185313
1: -0.0037615, 0.0007253, -0.0037847, 0.0008072, -0.0045687, 0.0045100
2: -0.0141255, 0.0098799, -0.0148080, 0.0100032, -0.0241287, 0.0246879
3: -0.0057700, 0.0050524, -0.0058261, 0.0052501, -0.0110201, 0.0108786
4: -0.0021620, 0.0024401, -0.0022460, 0.0024640, -0.0046259, 0.0046861
5: -0.0185200, 0.0113857, -0.0190661, 0.0115408, -0.0275353, 0.0284667
6: -0.0013490, 0.0066894, -0.0013883, 0.0073159, -0.0086649, 0.0080777
7: -0.0066279, 0.0130108, -0.0067297, 0.0133694, -0.0199973, 0.0197405
8: -0.0030497, 0.0072781, -0.0031032, 0.0074667, -0.0105164, 0.0103813
9: -0.0103032, 0.0016724, -0.0105219, 0.0017345, -0.0120377, 0.0121942

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126686, upper bound: 0.0135703
time: 2.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126686, upper bound: 0.0136213
time: 2.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9899768, 1.0084147, 0.9897113, 1.0100468, -0.0200700, 0.0187035
1: -0.0037615, 0.0007253, -0.0038276, 0.0008927, -0.0046542, 0.0045529
2: -0.0141255, 0.0098799, -0.0155202, 0.0102304, -0.0243559, 0.0254001
3: -0.0057700, 0.0050524, -0.0059296, 0.0054563, -0.0112263, 0.0109820
4: -0.0021620, 0.0024401, -0.0023337, 0.0025080, -0.0046699, 0.0047738
5: -0.0185200, 0.0113857, -0.0196360, 0.0118266, -0.0282485, 0.0291164
6: -0.0013490, 0.0066894, -0.0014609, 0.0079698, -0.0093188, 0.0081503
7: -0.0066279, 0.0130108, -0.0069174, 0.0137437, -0.0203715, 0.0199282
8: -0.0030497, 0.0072781, -0.0032019, 0.0076635, -0.0107132, 0.0104800
9: -0.0103032, 0.0016724, -0.0107501, 0.0018489, -0.0121521, 0.0124225

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126686, upper bound: 0.0136135
time: 2.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126686, upper bound: 0.0136723
time: 2.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0077457, 0.9899784, 1.0069005, -0.0175919, 0.0171198
1: -0.0039880, 0.0006661, -0.0037611, 0.0004555, -0.0043834, 0.0042658
2: -0.0135839, 0.0110805, -0.0124677, 0.0098777, -0.0226066, 0.0232299
3: -0.0063165, 0.0049097, -0.0057690, 0.0044017, -0.0105733, 0.0102895
4: -0.0021013, 0.0026725, -0.0018852, 0.0024397, -0.0043755, 0.0044961
5: -0.0181255, 0.0128958, -0.0167217, 0.0113830, -0.0284331, 0.0292172
6: -0.0017323, 0.0061413, -0.0013483, 0.0057850, -0.0074156, 0.0072166
7: -0.0076195, 0.0127518, -0.0066261, 0.0118299, -0.0191865, 0.0186716
8: -0.0035712, 0.0071419, -0.0030487, 0.0066571, -0.0100900, 0.0098192
9: -0.0101452, 0.0022771, -0.0095830, 0.0016713, -0.0113859, 0.0116998

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124438, upper bound: 0.0135480
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124438, upper bound: 0.0135576
time: 2.24 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0077457, 0.9898068, 1.0072820, -0.0180627, 0.0175122
1: -0.0039880, 0.0006661, -0.0038038, 0.0005505, -0.0045008, 0.0043636
2: -0.0135839, 0.0110805, -0.0129715, 0.0101043, -0.0231246, 0.0238516
3: -0.0063165, 0.0049097, -0.0058722, 0.0046309, -0.0108562, 0.0105253
4: -0.0021013, 0.0026725, -0.0019827, 0.0024836, -0.0044757, 0.0046164
5: -0.0181255, 0.0128958, -0.0173553, 0.0116680, -0.0290847, 0.0299991
6: -0.0017323, 0.0061413, -0.0014206, 0.0059458, -0.0076141, 0.0073820
7: -0.0076195, 0.0127518, -0.0068132, 0.0122459, -0.0197000, 0.0190995
8: -0.0035712, 0.0071419, -0.0031472, 0.0068759, -0.0103600, 0.0100442
9: -0.0101452, 0.0022771, -0.0098368, 0.0017854, -0.0116468, 0.0120129

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124438, upper bound: 0.0135621
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124438, upper bound: 0.0135728
time: 2.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9898834, 1.0092132, -0.0201457, 0.0179543
1: -0.0039880, 0.0006661, -0.0037847, 0.0008072, -0.0047952, 0.0044508
2: -0.0136325, 0.0110805, -0.0148080, 0.0100032, -0.0236357, 0.0258885
3: -0.0063165, 0.0049097, -0.0058261, 0.0052501, -0.0115665, 0.0107358
4: -0.0021013, 0.0026725, -0.0022460, 0.0024640, -0.0045652, 0.0049185
5: -0.0181255, 0.0128958, -0.0190661, 0.0115408, -0.0273693, 0.0300272
6: -0.0017323, 0.0062368, -0.0013883, 0.0073159, -0.0090482, 0.0076251
7: -0.0076195, 0.0127518, -0.0067297, 0.0133694, -0.0209889, 0.0194814
8: -0.0035712, 0.0071419, -0.0031032, 0.0074667, -0.0110379, 0.0102451
9: -0.0101452, 0.0022771, -0.0105219, 0.0017345, -0.0118797, 0.0127989

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126366, upper bound: 0.0136614
time: 2.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126367, upper bound: 0.0137000
time: 2.74 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.60 + 599.29 = 604.89 seconds
