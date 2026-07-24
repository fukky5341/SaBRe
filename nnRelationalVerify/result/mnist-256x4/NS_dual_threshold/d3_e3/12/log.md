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
execution time: IAR + RelationalAnalysis = 0.77 + 3.25 = 4.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0149071, upper bound: 0.0149071

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145435, upper bound: 0.0145898
time: 1.79 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145898, upper bound: 0.0145898
time: 2.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.14 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.14
Output dim: 0, lower bound: -0.0145435, upper bound: 0.0145898
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.14
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

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141317, upper bound: 0.0140801
time: 2.18 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141317, upper bound: 0.0141643
time: 2.10 seconds

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

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141648, upper bound: 0.0140802
time: 2.47 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0141648, upper bound: 0.0141648
time: 2.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.40 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 0, lower bound: -0.0141317, upper bound: 0.0140801
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 0, lower bound: -0.0141317, upper bound: 0.0141643
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 0, lower bound: -0.0141648, upper bound: 0.0140802
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 0, lower bound: -0.0141648, upper bound: 0.0141648

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.9897838, 1.0099065, 0.9890434, 1.0113680, -0.0215842, 0.0208631
1: -0.0038096, 0.0008783, -0.0039940, 0.0010282, -0.0048378, 0.0048723
2: -0.0154003, 0.0101347, -0.0166492, 0.0111122, -0.0265126, 0.0267839
3: -0.0058860, 0.0054216, -0.0063309, 0.0057832, -0.0116692, 0.0117525
4: -0.0023189, 0.0024894, -0.0024727, 0.0026786, -0.0049976, 0.0049621
5: -0.0195401, 0.0117062, -0.0205394, 0.0129357, -0.0316814, 0.0314454
6: -0.0014303, 0.0078597, -0.0017424, 0.0090062, -0.0104366, 0.0096021
7: -0.0068383, 0.0136807, -0.0076457, 0.0143369, -0.0211752, 0.0213264
8: -0.0031604, 0.0076304, -0.0035850, 0.0079755, -0.0111358, 0.0112153
9: -0.0107116, 0.0018007, -0.0111118, 0.0022931, -0.0130047, 0.0129126

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130432, upper bound: 0.0137987
time: 1.84 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139947, upper bound: 0.0139280
time: 2.19 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.9888838, 1.0094743, 0.9891217, 1.0106951, -0.0218113, 0.0203525
1: -0.0040338, 0.0008340, -0.0039745, 0.0009592, -0.0049930, 0.0048085
2: -0.0150309, 0.0113232, -0.0160741, 0.0110090, -0.0260399, 0.0273973
3: -0.0064269, 0.0053146, -0.0062839, 0.0056167, -0.0120436, 0.0115985
4: -0.0022734, 0.0027195, -0.0024019, 0.0026586, -0.0049321, 0.0051213
5: -0.0192445, 0.0132010, -0.0200792, 0.0128058, -0.0318033, 0.0323887
6: -0.0018097, 0.0075206, -0.0017094, 0.0084782, -0.0102879, 0.0092300
7: -0.0078200, 0.0134865, -0.0075604, 0.0140347, -0.0218546, 0.0210470
8: -0.0036766, 0.0075283, -0.0035401, 0.0078166, -0.0114931, 0.0110684
9: -0.0105933, 0.0023993, -0.0109275, 0.0022411, -0.0128343, 0.0133269

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130432, upper bound: 0.0138889
time: 2.13 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0139947, upper bound: 0.0140236
time: 1.95 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 0.9896037, 1.0108711, 0.9890260, 1.0113904, -0.0217867, 0.0218450
1: -0.0038544, 0.0009772, -0.0039984, 0.0010305, -0.0048850, 0.0049756
2: -0.0162245, 0.0103725, -0.0166683, 0.0111354, -0.0273599, 0.0270408
3: -0.0059942, 0.0056602, -0.0063415, 0.0057888, -0.0117830, 0.0120017
4: -0.0024204, 0.0025355, -0.0024751, 0.0026831, -0.0051035, 0.0050105
5: -0.0201995, 0.0120053, -0.0205547, 0.0129649, -0.0323783, 0.0318589
6: -0.0015062, 0.0086163, -0.0017498, 0.0090238, -0.0105300, 0.0103661
7: -0.0070347, 0.0141137, -0.0076649, 0.0143469, -0.0213817, 0.0217786
8: -0.0032636, 0.0078581, -0.0035950, 0.0079808, -0.0112444, 0.0114531
9: -0.0109757, 0.0019205, -0.0111179, 0.0023047, -0.0132805, 0.0130384

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0137988
time: 1.87 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140246, upper bound: 0.0139280
time: 2.17 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 0.9888224, 1.0101365, 0.9891040, 1.0107176, -0.0218952, 0.0210325
1: -0.0040491, 0.0009019, -0.0039789, 0.0009615, -0.0050106, 0.0048808
2: -0.0155967, 0.0114041, -0.0160933, 0.0110323, -0.0266291, 0.0274974
3: -0.0064638, 0.0054785, -0.0062946, 0.0056222, -0.0120860, 0.0117730
4: -0.0023431, 0.0027351, -0.0024043, 0.0026632, -0.0050063, 0.0051394
5: -0.0196972, 0.0133028, -0.0200946, 0.0128352, -0.0324458, 0.0328464
6: -0.0018356, 0.0080400, -0.0017169, 0.0084959, -0.0103314, 0.0097569
7: -0.0078868, 0.0137839, -0.0075797, 0.0140448, -0.0219316, 0.0213636
8: -0.0037117, 0.0076846, -0.0035502, 0.0078219, -0.0115336, 0.0112349
9: -0.0107746, 0.0024401, -0.0109337, 0.0022528, -0.0130274, 0.0133738

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0138892
time: 1.91 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0140246, upper bound: 0.0140246
time: 2.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.85 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0130432, upper bound: 0.0137987
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0139947, upper bound: 0.0139280
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0130432, upper bound: 0.0138889
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0139947, upper bound: 0.0140236
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0137988
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0140246, upper bound: 0.0139280
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0130940, upper bound: 0.0138892
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 0, lower bound: -0.0140246, upper bound: 0.0140246

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9898740, 1.0086151, 0.9893060, 1.0075849, -0.0175174, 0.0193092
1: -0.0037871, 0.0007458, -0.0039286, 0.0006260, -0.0044131, 0.0046540
2: -0.0142968, 0.0100156, -0.0133716, 0.0107656, -0.0250623, 0.0231315
3: -0.0058318, 0.0051020, -0.0061731, 0.0048130, -0.0106448, 0.0112259
4: -0.0021830, 0.0024664, -0.0020601, 0.0026115, -0.0047736, 0.0045265
5: -0.0186571, 0.0115564, -0.0178584, 0.0124997, -0.0310206, 0.0283323
6: -0.0013923, 0.0068466, -0.0016317, 0.0060735, -0.0073842, 0.0084783
7: -0.0067400, 0.0131008, -0.0073594, 0.0125764, -0.0193163, 0.0203708
8: -0.0031086, 0.0073254, -0.0034344, 0.0070496, -0.0101583, 0.0107128
9: -0.0103581, 0.0017407, -0.0100383, 0.0021185, -0.0124220, 0.0117790

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0131707
time: 2.01 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0134956
time: 2.34 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9897850, 1.0098994, 0.9891220, 1.0108925, -0.0211076, 0.0207774
1: -0.0038093, 0.0008776, -0.0039745, 0.0009795, -0.0047887, 0.0048520
2: -0.0153942, 0.0101332, -0.0162429, 0.0110085, -0.0264027, 0.0263761
3: -0.0058853, 0.0054198, -0.0062837, 0.0056656, -0.0115509, 0.0117035
4: -0.0023182, 0.0024892, -0.0024227, 0.0026586, -0.0049767, 0.0049118
5: -0.0195351, 0.0117043, -0.0202143, 0.0128053, -0.0315490, 0.0302331
6: -0.0014299, 0.0078540, -0.0017093, 0.0086332, -0.0100631, 0.0095633
7: -0.0068371, 0.0136774, -0.0075601, 0.0141234, -0.0209605, 0.0212375
8: -0.0031597, 0.0076287, -0.0035399, 0.0078632, -0.0110229, 0.0111686
9: -0.0107097, 0.0018000, -0.0109816, 0.0022408, -0.0129505, 0.0127816

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136614, upper bound: 0.0132518
time: 2.25 seconds

## Relational analysis of NS_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137001, upper bound: 0.0136378
time: 2.16 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889625, 1.0082163, 0.9893630, 1.0073020, -0.0181977, 0.0188533
1: -0.0040142, 0.0007049, -0.0039144, 0.0005555, -0.0045697, 0.0046193
2: -0.0139559, 0.0112191, -0.0129979, 0.0106904, -0.0246463, 0.0240299
3: -0.0063796, 0.0050033, -0.0061389, 0.0046429, -0.0110225, 0.0111422
4: -0.0021411, 0.0026993, -0.0019878, 0.0025970, -0.0047381, 0.0046872
5: -0.0183843, 0.0130701, -0.0173885, 0.0124051, -0.0307894, 0.0294640
6: -0.0017765, 0.0065337, -0.0016077, 0.0059542, -0.0076710, 0.0081414
7: -0.0077340, 0.0129217, -0.0072973, 0.0122677, -0.0200017, 0.0202190
8: -0.0036314, 0.0072312, -0.0034017, 0.0068873, -0.0105187, 0.0106330
9: -0.0102488, 0.0023469, -0.0098500, 0.0020806, -0.0123294, 0.0121970

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0132927
time: 2.03 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
time: 2.07 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888847, 1.0094664, 0.9892001, 1.0102011, -0.0213163, 0.0202663
1: -0.0040336, 0.0008332, -0.0039550, 0.0009085, -0.0049421, 0.0047881
2: -0.0150242, 0.0113218, -0.0156521, 0.0109054, -0.0259296, 0.0269739
3: -0.0064263, 0.0053127, -0.0062368, 0.0054945, -0.0119208, 0.0115494
4: -0.0022726, 0.0027192, -0.0023499, 0.0026386, -0.0049112, 0.0050691
5: -0.0192391, 0.0131993, -0.0197415, 0.0126755, -0.0316710, 0.0310337
6: -0.0018093, 0.0075144, -0.0016764, 0.0080908, -0.0099001, 0.0091907
7: -0.0078188, 0.0134830, -0.0074749, 0.0138129, -0.0216317, 0.0209579
8: -0.0036760, 0.0075264, -0.0034951, 0.0076999, -0.0113759, 0.0110215
9: -0.0105911, 0.0023986, -0.0107923, 0.0021889, -0.0127800, 0.0131909

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136614, upper bound: 0.0133805
time: 2.34 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137001, upper bound: 0.0137193
time: 2.16 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9897105, 1.0096176, 0.9892781, 1.0075934, -0.0178829, 0.0203394
1: -0.0038278, 0.0008487, -0.0039356, 0.0006281, -0.0044559, 0.0047842
2: -0.0151534, 0.0102315, -0.0133826, 0.0108024, -0.0259558, 0.0236141
3: -0.0059300, 0.0053501, -0.0061899, 0.0048181, -0.0107481, 0.0115400
4: -0.0022885, 0.0025082, -0.0020623, 0.0026187, -0.0049072, 0.0045705
5: -0.0193425, 0.0118279, -0.0178724, 0.0125460, -0.0318885, 0.0288833
6: -0.0014612, 0.0076330, -0.0016435, 0.0060770, -0.0075383, 0.0092765
7: -0.0069183, 0.0135509, -0.0073898, 0.0125855, -0.0195038, 0.0209407
8: -0.0032024, 0.0075622, -0.0034504, 0.0070545, -0.0102568, 0.0110125
9: -0.0106325, 0.0018495, -0.0100438, 0.0021370, -0.0127695, 0.0118933

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0131707
time: 2.28 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0134956
time: 2.15 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9896049, 1.0108631, 0.9891049, 1.0109150, -0.0213101, 0.0217581
1: -0.0038541, 0.0009764, -0.0039787, 0.0009817, -0.0048359, 0.0049551
2: -0.0162176, 0.0103709, -0.0162620, 0.0110311, -0.0272488, 0.0266329
3: -0.0059935, 0.0056582, -0.0062940, 0.0056711, -0.0116646, 0.0119523
4: -0.0024196, 0.0025352, -0.0024250, 0.0026629, -0.0050825, 0.0049602
5: -0.0201940, 0.0120033, -0.0202296, 0.0128337, -0.0322451, 0.0306147
6: -0.0015057, 0.0086100, -0.0017165, 0.0086508, -0.0101565, 0.0103265
7: -0.0070334, 0.0141101, -0.0075787, 0.0141334, -0.0211669, 0.0216888
8: -0.0032629, 0.0078562, -0.0035497, 0.0078685, -0.0111314, 0.0114059
9: -0.0109735, 0.0019197, -0.0109877, 0.0022522, -0.0132257, 0.0129075

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136706, upper bound: 0.0132518
time: 2.10 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137204, upper bound: 0.0136379
time: 2.36 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9889289, 1.0088717, 0.9893404, 1.0073105, -0.0183817, 0.0195312
1: -0.0040226, 0.0007722, -0.0039200, 0.0005576, -0.0045802, 0.0046922
2: -0.0145161, 0.0112636, -0.0130091, 0.0107202, -0.0252363, 0.0242728
3: -0.0063998, 0.0051655, -0.0061525, 0.0046481, -0.0110479, 0.0113180
4: -0.0022101, 0.0027079, -0.0019900, 0.0026028, -0.0048128, 0.0046979
5: -0.0188326, 0.0131261, -0.0174026, 0.0124426, -0.0312752, 0.0298837
6: -0.0017907, 0.0070480, -0.0016172, 0.0059578, -0.0077485, 0.0086652
7: -0.0077708, 0.0132161, -0.0073219, 0.0122770, -0.0200478, 0.0205380
8: -0.0036507, 0.0073860, -0.0034147, 0.0068922, -0.0105429, 0.0108007
9: -0.0104283, 0.0023693, -0.0098557, 0.0020956, -0.0125240, 0.0122251

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126844, upper bound: 0.0132927
time: 2.36 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135736
time: 2.19 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9888235, 1.0101287, 0.9891830, 1.0102239, -0.0214003, 0.0209457
1: -0.0040488, 0.0009011, -0.0039593, 0.0009108, -0.0049597, 0.0048604
2: -0.0155902, 0.0114026, -0.0156714, 0.0109281, -0.0265183, 0.0270740
3: -0.0064631, 0.0054766, -0.0062471, 0.0055001, -0.0119632, 0.0117237
4: -0.0023423, 0.0027348, -0.0023523, 0.0026430, -0.0049853, 0.0050871
5: -0.0196920, 0.0133009, -0.0197570, 0.0127041, -0.0323127, 0.0315932
6: -0.0018351, 0.0080340, -0.0016836, 0.0081085, -0.0099436, 0.0097176
7: -0.0078855, 0.0137804, -0.0074936, 0.0138231, -0.0217086, 0.0212741
8: -0.0037111, 0.0076828, -0.0035050, 0.0077053, -0.0114163, 0.0111878
9: -0.0107725, 0.0024393, -0.0107985, 0.0022003, -0.0129728, 0.0132378

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136706, upper bound: 0.0133805
time: 2.23 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0137203, upper bound: 0.0137204
time: 2.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.45 seconds
NS_A1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0131707
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0134956
NS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0136614, upper bound: 0.0132518
NS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0137001, upper bound: 0.0136378
NS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0132927
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0136614, upper bound: 0.0133805
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0137001, upper bound: 0.0137193
NS_A2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0131707
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0134956
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0136706, upper bound: 0.0132518
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0137204, upper bound: 0.0136379
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0126844, upper bound: 0.0132927
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135736
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0136706, upper bound: 0.0133805
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -0.0137203, upper bound: 0.0137204

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9893304, 1.0074903, -0.0172999, 0.0177612
1: -0.0037574, 0.0006411, -0.0039225, 0.0006024, -0.0043107, 0.0044256
2: -0.0134516, 0.0098585, -0.0132465, 0.0107333, -0.0234535, 0.0228443
3: -0.0057603, 0.0048494, -0.0061585, 0.0047561, -0.0103977, 0.0106750
4: -0.0020756, 0.0024360, -0.0020359, 0.0026053, -0.0045394, 0.0044215
5: -0.0179591, 0.0113588, -0.0177012, 0.0124592, -0.0294984, 0.0287321
6: -0.0013422, 0.0060990, -0.0016214, 0.0060336, -0.0072925, 0.0074870
7: -0.0066102, 0.0126424, -0.0073328, 0.0124731, -0.0188680, 0.0193712
8: -0.0030404, 0.0070844, -0.0034204, 0.0069953, -0.0099225, 0.0101871
9: -0.0100785, 0.0016616, -0.0099753, 0.0021022, -0.0118124, 0.0115056

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
time: 2.03 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
time: 2.21 seconds

## BFS NS instance: NS_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9903163, 1.0073907, 0.9892128, 1.0099665, -0.0196502, 0.0181779
1: -0.0036769, 0.0005776, -0.0039518, 0.0008845, -0.0043210, 0.0045295
2: -0.0131151, 0.0094314, -0.0154517, 0.0108887, -0.0240037, 0.0248831
3: -0.0055659, 0.0046963, -0.0062292, 0.0054364, -0.0104226, 0.0109255
4: -0.0020105, 0.0023533, -0.0023252, 0.0026354, -0.0046459, 0.0044320
5: -0.0175359, 0.0108217, -0.0195811, 0.0126545, -0.0293879, 0.0288007
6: -0.0012058, 0.0059916, -0.0016710, 0.0079068, -0.0091126, 0.0076626
7: -0.0062574, 0.0123645, -0.0074611, 0.0137076, -0.0189130, 0.0198256
8: -0.0028549, 0.0069382, -0.0034878, 0.0076446, -0.0099462, 0.0104261
9: -0.0099091, 0.0014465, -0.0107281, 0.0021805, -0.0120896, 0.0115331

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135426, upper bound: 0.0132519
time: 2.53 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135426, upper bound: 0.0132518
time: 2.37 seconds

## BFS NS instance: NS_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9899105, 1.0088818, 0.9891485, 1.0106665, -0.0207559, 0.0197333
1: -0.0037780, 0.0007732, -0.0039679, 0.0009563, -0.0047342, 0.0047410
2: -0.0145245, 0.0099673, -0.0160497, 0.0109735, -0.0254981, 0.0260170
3: -0.0058098, 0.0051680, -0.0062678, 0.0056096, -0.0114194, 0.0114358
4: -0.0022111, 0.0024570, -0.0023989, 0.0026518, -0.0048629, 0.0048559
5: -0.0188393, 0.0114956, -0.0200597, 0.0127613, -0.0302388, 0.0295674
6: -0.0013769, 0.0070557, -0.0016981, 0.0084558, -0.0098327, 0.0087538
7: -0.0067000, 0.0132205, -0.0075312, 0.0140219, -0.0207219, 0.0207516
8: -0.0030876, 0.0073884, -0.0035247, 0.0078098, -0.0108975, 0.0109131
9: -0.0104310, 0.0017164, -0.0109197, 0.0022232, -0.0126543, 0.0126361

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135878, upper bound: 0.0136378
time: 2.15 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135878, upper bound: 0.0136378
time: 2.16 seconds

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9893860, 1.0072069, -0.0179733, 0.0180559
1: -0.0039844, 0.0005904, -0.0039087, 0.0005318, -0.0044785, 0.0044990
2: -0.0131826, 0.0110613, -0.0128723, 0.0106600, -0.0238426, 0.0237336
3: -0.0063077, 0.0047270, -0.0061251, 0.0045858, -0.0108025, 0.0108521
4: -0.0020236, 0.0026688, -0.0019635, 0.0025911, -0.0046147, 0.0045936
5: -0.0176208, 0.0128716, -0.0172305, 0.0123669, -0.0299878, 0.0298506
6: -0.0017261, 0.0060132, -0.0015980, 0.0059141, -0.0075764, 0.0076112
7: -0.0076036, 0.0124203, -0.0072722, 0.0121640, -0.0196024, 0.0196925
8: -0.0035628, 0.0069676, -0.0033885, 0.0068328, -0.0103087, 0.0103561
9: -0.0099431, 0.0022674, -0.0097868, 0.0020653, -0.0120084, 0.0119535

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135501
time: 2.32 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
time: 2.28 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9893407, 1.0072719, 0.9892923, 1.0092841, -0.0199434, 0.0179796
1: -0.0039200, 0.0005480, -0.0039320, 0.0008145, -0.0045007, 0.0044801
2: -0.0129582, 0.0107198, -0.0148684, 0.0107837, -0.0237420, 0.0255882
3: -0.0061523, 0.0046249, -0.0061814, 0.0052676, -0.0108561, 0.0108063
4: -0.0019801, 0.0026027, -0.0022534, 0.0026151, -0.0045952, 0.0046164
5: -0.0173386, 0.0124421, -0.0191145, 0.0125226, -0.0295432, 0.0299986
6: -0.0016171, 0.0059415, -0.0016375, 0.0073714, -0.0089885, 0.0075791
7: -0.0073216, 0.0122350, -0.0073744, 0.0134012, -0.0196996, 0.0196094
8: -0.0034145, 0.0068701, -0.0034423, 0.0074834, -0.0103598, 0.0103124
9: -0.0098301, 0.0020954, -0.0105412, 0.0021276, -0.0119577, 0.0120128

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_A2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0126032
time: 2.55 seconds

## Relational analysis of NS_A1_A2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0133805
time: 1.94 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9890066, 1.0083488, 0.9892271, 1.0099709, -0.0209643, 0.0191218
1: -0.0040032, 0.0007185, -0.0039483, 0.0008849, -0.0048881, 0.0046668
2: -0.0140692, 0.0111609, -0.0154553, 0.0108699, -0.0249391, 0.0266162
3: -0.0063531, 0.0050361, -0.0062206, 0.0054375, -0.0117906, 0.0112568
4: -0.0021550, 0.0026881, -0.0023257, 0.0026317, -0.0047867, 0.0050138
5: -0.0184750, 0.0129970, -0.0195840, 0.0126309, -0.0303217, 0.0306734
6: -0.0017579, 0.0066377, -0.0016650, 0.0079101, -0.0096681, 0.0083027
7: -0.0076860, 0.0129812, -0.0074455, 0.0137095, -0.0213955, 0.0204267
8: -0.0036061, 0.0072625, -0.0034797, 0.0076456, -0.0112517, 0.0107422
9: -0.0102851, 0.0023176, -0.0107293, 0.0021710, -0.0124561, 0.0130469

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126973, upper bound: 0.0127323
time: 2.13 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126973, upper bound: 0.0137193
time: 2.26 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9893027, 1.0074987, -0.0176567, 0.0192495
1: -0.0037951, 0.0007394, -0.0039294, 0.0006045, -0.0043996, 0.0045879
2: -0.0142430, 0.0100580, -0.0132577, 0.0107700, -0.0250130, 0.0233156
3: -0.0058511, 0.0050865, -0.0061751, 0.0047612, -0.0106123, 0.0110665
4: -0.0021764, 0.0024746, -0.0020381, 0.0026124, -0.0047059, 0.0045127
5: -0.0186140, 0.0116097, -0.0177152, 0.0125053, -0.0305802, 0.0285072
6: -0.0014058, 0.0067972, -0.0016331, 0.0060371, -0.0074430, 0.0084304
7: -0.0067750, 0.0130725, -0.0073630, 0.0124823, -0.0192573, 0.0200816
8: -0.0031270, 0.0073106, -0.0034363, 0.0070002, -0.0101272, 0.0105607
9: -0.0103408, 0.0017621, -0.0099809, 0.0021207, -0.0122457, 0.0117430

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126322, upper bound: 0.0134956
time: 2.39 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126322, upper bound: 0.0134956
time: 2.14 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9900842, 1.0080066, 0.9891961, 1.0099885, -0.0199043, 0.0188105
1: -0.0037347, 0.0006834, -0.0039560, 0.0008867, -0.0046214, 0.0046394
2: -0.0137768, 0.0097380, -0.0154703, 0.0109107, -0.0246875, 0.0252083
3: -0.0057054, 0.0049515, -0.0062392, 0.0054418, -0.0111473, 0.0111907
4: -0.0021190, 0.0024127, -0.0023275, 0.0026396, -0.0047586, 0.0047402
5: -0.0182410, 0.0112073, -0.0195961, 0.0126823, -0.0302336, 0.0291725
6: -0.0013037, 0.0063692, -0.0016781, 0.0079239, -0.0092276, 0.0080473
7: -0.0065107, 0.0128275, -0.0074793, 0.0137174, -0.0202281, 0.0203068
8: -0.0029880, 0.0071817, -0.0034974, 0.0076497, -0.0106378, 0.0106792
9: -0.0101914, 0.0016009, -0.0107341, 0.0021916, -0.0123830, 0.0123350

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135848, upper bound: 0.0132518
time: 2.35 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0135848, upper bound: 0.0132518
time: 2.57 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9897384, 1.0098001, 0.9891315, 1.0106889, -0.0209505, 0.0206686
1: -0.0038209, 0.0008674, -0.0039721, 0.0009586, -0.0047794, 0.0048395
2: -0.0153093, 0.0101947, -0.0160688, 0.0109961, -0.0263054, 0.0262635
3: -0.0059133, 0.0053952, -0.0062781, 0.0056152, -0.0115285, 0.0116733
4: -0.0023077, 0.0025010, -0.0024012, 0.0026562, -0.0049639, 0.0049023
5: -0.0194672, 0.0117816, -0.0200750, 0.0127896, -0.0309873, 0.0302356
6: -0.0014495, 0.0077761, -0.0017053, 0.0084734, -0.0099229, 0.0094814
7: -0.0068878, 0.0136328, -0.0075498, 0.0140319, -0.0209198, 0.0211826
8: -0.0031864, 0.0076052, -0.0035345, 0.0078151, -0.0110015, 0.0111397
9: -0.0106825, 0.0018309, -0.0109259, 0.0022346, -0.0129171, 0.0127568

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136384, upper bound: 0.0136379
time: 2.03 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0136384, upper bound: 0.0136379
time: 2.12 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9893652, 1.0072154, -0.0181565, 0.0183900
1: -0.0039902, 0.0006576, -0.0039139, 0.0005339, -0.0045241, 0.0045715
2: -0.0135619, 0.0110920, -0.0128836, 0.0106875, -0.0242494, 0.0239756
3: -0.0063217, 0.0048892, -0.0061376, 0.0045909, -0.0109127, 0.0110268
4: -0.0020926, 0.0026747, -0.0019657, 0.0025964, -0.0046890, 0.0046404
5: -0.0180690, 0.0129103, -0.0172447, 0.0124015, -0.0304705, 0.0295053
6: -0.0017359, 0.0061720, -0.0016068, 0.0059177, -0.0076537, 0.0077788
7: -0.0076290, 0.0127146, -0.0072949, 0.0121733, -0.0198023, 0.0200095
8: -0.0035762, 0.0071224, -0.0034005, 0.0068377, -0.0104139, 0.0105228
9: -0.0101226, 0.0022829, -0.0097925, 0.0020791, -0.0122017, 0.0120754

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135501
time: 2.38 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135536
time: 2.29 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9892592, 1.0075455, 0.9892755, 1.0093062, -0.0200469, 0.0182699
1: -0.0039403, 0.0006162, -0.0039362, 0.0008167, -0.0046587, 0.0045524
2: -0.0133193, 0.0108273, -0.0148873, 0.0108059, -0.0241252, 0.0257146
3: -0.0062012, 0.0047893, -0.0061915, 0.0052730, -0.0112372, 0.0109808
4: -0.0020500, 0.0026235, -0.0022558, 0.0026194, -0.0046694, 0.0047784
5: -0.0177927, 0.0125773, -0.0191296, 0.0125504, -0.0302829, 0.0310518
6: -0.0016514, 0.0060568, -0.0016446, 0.0073888, -0.0090402, 0.0077014
7: -0.0074104, 0.0125332, -0.0073927, 0.0134111, -0.0203913, 0.0199259
8: -0.0034612, 0.0070269, -0.0034519, 0.0074886, -0.0107236, 0.0104789
9: -0.0100119, 0.0021496, -0.0105473, 0.0021388, -0.0121507, 0.0124345

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126844, upper bound: 0.0126033
time: 2.24 seconds

## Relational analysis of NS_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126844, upper bound: 0.0133805
time: 2.17 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9889547, 1.0090044, 0.9892098, 1.0099934, -0.0210388, 0.0197945
1: -0.0040162, 0.0007857, -0.0039526, 0.0008872, -0.0049034, 0.0047383
2: -0.0146293, 0.0112295, -0.0154746, 0.0108926, -0.0255219, 0.0267041
3: -0.0063843, 0.0051983, -0.0062310, 0.0054431, -0.0118274, 0.0114293
4: -0.0022240, 0.0027013, -0.0023281, 0.0026361, -0.0048601, 0.0050294
5: -0.0189231, 0.0130833, -0.0195995, 0.0126595, -0.0310086, 0.0312148
6: -0.0017798, 0.0071519, -0.0016723, 0.0079279, -0.0097077, 0.0088242
7: -0.0077426, 0.0132755, -0.0074643, 0.0137197, -0.0214623, 0.0207398
8: -0.0036359, 0.0074173, -0.0034896, 0.0076509, -0.0112868, 0.0109069
9: -0.0104646, 0.0023522, -0.0107354, 0.0021825, -0.0126471, 0.0130876

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0127323
time: 2.42 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0137204
time: 2.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.30 seconds
NS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
NS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
NS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0135426, upper bound: 0.0132519
NS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0135426, upper bound: 0.0132518
NS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0135878, upper bound: 0.0136378
NS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0135878, upper bound: 0.0136378
NS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135501
NS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135728
NS_A1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0126032
NS_A1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126950, upper bound: 0.0133805
NS_A1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126973, upper bound: 0.0127323
NS_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126973, upper bound: 0.0137193
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126322, upper bound: 0.0134956
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126322, upper bound: 0.0134956
NS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0135848, upper bound: 0.0132518
NS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0135848, upper bound: 0.0132518
NS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0136384, upper bound: 0.0136379
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0136384, upper bound: 0.0136379
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135501
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135536
NS_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126844, upper bound: 0.0126033
NS_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0126844, upper bound: 0.0133805
NS_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0127323
NS_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.30
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0137204

## BFS NS instance: NS_A1_A1_B1_A2_B1

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

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134333
time: 2.06 seconds

## Relational analysis of NS_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
time: 2.00 seconds

## BFS NS instance: NS_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9886200, 1.0068272, -0.0166946, 0.0182548
1: -0.0037574, 0.0006411, -0.0040995, 0.0004372, -0.0041598, 0.0045486
2: -0.0134516, 0.0098585, -0.0123709, 0.0116715, -0.0241053, 0.0220450
3: -0.0057603, 0.0048494, -0.0065855, 0.0043576, -0.0100339, 0.0109717
4: -0.0020756, 0.0024360, -0.0018665, 0.0027869, -0.0046655, 0.0042668
5: -0.0179591, 0.0113588, -0.0165999, 0.0136391, -0.0303181, 0.0277268
6: -0.0013422, 0.0060990, -0.0019209, 0.0057541, -0.0070374, 0.0076951
7: -0.0066102, 0.0126424, -0.0081076, 0.0117499, -0.0182078, 0.0199094
8: -0.0030404, 0.0070844, -0.0038279, 0.0066150, -0.0095753, 0.0104702
9: -0.0100785, 0.0016616, -0.0095343, 0.0025747, -0.0121407, 0.0111030

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134333
time: 2.31 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
time: 2.26 seconds

## BFS NS instance: NS_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9903163, 1.0073907, 0.9893863, 1.0087494, -0.0184330, 0.0180045
1: -0.0036769, 0.0005776, -0.0039086, 0.0007596, -0.0041708, 0.0044862
2: -0.0131151, 0.0094314, -0.0144116, 0.0106597, -0.0237747, 0.0238430
3: -0.0055659, 0.0046963, -0.0061249, 0.0051353, -0.0100603, 0.0108212
4: -0.0020105, 0.0023533, -0.0021972, 0.0025910, -0.0046016, 0.0042780
5: -0.0175359, 0.0108217, -0.0187489, 0.0123665, -0.0291009, 0.0277997
6: -0.0012058, 0.0059916, -0.0015979, 0.0069520, -0.0081578, 0.0075895
7: -0.0062574, 0.0123645, -0.0072719, 0.0131611, -0.0182557, 0.0196364
8: -0.0028549, 0.0069382, -0.0033884, 0.0073572, -0.0096005, 0.0103266
9: -0.0099091, 0.0014465, -0.0103948, 0.0020651, -0.0119742, 0.0111322

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0125735
time: 2.31 seconds

## Relational analysis of NS_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0132519
time: 2.14 seconds

## BFS NS instance: NS_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9903163, 1.0073907, 0.9885150, 1.0083014, -0.0179850, 0.0188758
1: -0.0036769, 0.0005776, -0.0041257, 0.0007137, -0.0041652, 0.0047033
2: -0.0131151, 0.0094314, -0.0140288, 0.0118102, -0.0249253, 0.0234602
3: -0.0055659, 0.0046963, -0.0066486, 0.0050244, -0.0100468, 0.0113449
4: -0.0020105, 0.0023533, -0.0021500, 0.0028137, -0.0048242, 0.0042722
5: -0.0175359, 0.0108217, -0.0184426, 0.0138136, -0.0305913, 0.0277624
6: -0.0012058, 0.0059916, -0.0019652, 0.0066006, -0.0078064, 0.0079568
7: -0.0062574, 0.0123645, -0.0082222, 0.0129600, -0.0182311, 0.0205867
8: -0.0028549, 0.0069382, -0.0038881, 0.0072514, -0.0095876, 0.0108263
9: -0.0099091, 0.0014465, -0.0102722, 0.0026446, -0.0125537, 0.0111173

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0125735
time: 2.23 seconds

## Relational analysis of NS_A1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0132518
time: 2.09 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9899105, 1.0088818, 0.9893202, 1.0094391, -0.0195286, 0.0195616
1: -0.0037780, 0.0007732, -0.0039251, 0.0008303, -0.0046083, 0.0046982
2: -0.0145245, 0.0099673, -0.0150009, 0.0107468, -0.0252714, 0.0249681
3: -0.0058098, 0.0051680, -0.0061646, 0.0053059, -0.0111157, 0.0113326
4: -0.0022111, 0.0024570, -0.0022697, 0.0026079, -0.0048190, 0.0047268
5: -0.0188393, 0.0114956, -0.0192204, 0.0124761, -0.0299488, 0.0287257
6: -0.0013769, 0.0070557, -0.0016257, 0.0074930, -0.0088699, 0.0086814
7: -0.0067000, 0.0132205, -0.0073439, 0.0134708, -0.0201708, 0.0205644
8: -0.0030876, 0.0073884, -0.0034262, 0.0075200, -0.0106076, 0.0108146
9: -0.0104310, 0.0017164, -0.0105837, 0.0021090, -0.0125401, 0.0123001

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0127340
time: 2.25 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0136378
time: 2.30 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9899105, 1.0088818, 0.9884507, 1.0089504, -0.0190398, 0.0204311
1: -0.0037780, 0.0007732, -0.0041417, 0.0007802, -0.0045582, 0.0049149
2: -0.0145245, 0.0099673, -0.0145832, 0.0118950, -0.0264195, 0.0245504
3: -0.0058098, 0.0051680, -0.0066872, 0.0051850, -0.0109948, 0.0118552
4: -0.0022111, 0.0024570, -0.0022183, 0.0028301, -0.0050412, 0.0046753
5: -0.0188393, 0.0114956, -0.0188862, 0.0139202, -0.0314605, 0.0285399
6: -0.0013769, 0.0070557, -0.0019923, 0.0071095, -0.0084864, 0.0090480
7: -0.0067000, 0.0132205, -0.0082922, 0.0132513, -0.0199513, 0.0215127
8: -0.0030876, 0.0073884, -0.0039249, 0.0074046, -0.0104922, 0.0113133
9: -0.0104310, 0.0017164, -0.0104498, 0.0026873, -0.0131184, 0.0121662

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0127340
time: 2.29 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0136378
time: 2.16 seconds

## BFS NS instance: NS_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9899159, 1.0071138, -0.0178954, 0.0175238
1: -0.0039844, 0.0005904, -0.0037766, 0.0005086, -0.0044590, 0.0043665
2: -0.0131826, 0.0110613, -0.0127494, 0.0099602, -0.0231399, 0.0236306
3: -0.0063077, 0.0047270, -0.0058066, 0.0045299, -0.0107556, 0.0105323
4: -0.0020236, 0.0026688, -0.0019397, 0.0024557, -0.0044787, 0.0045737
5: -0.0176208, 0.0128716, -0.0170760, 0.0114867, -0.0291039, 0.0297211
6: -0.0017261, 0.0060132, -0.0013746, 0.0058749, -0.0075435, 0.0073869
7: -0.0076036, 0.0124203, -0.0066942, 0.0120625, -0.0195174, 0.0191121
8: -0.0035628, 0.0069676, -0.0030846, 0.0067794, -0.0102640, 0.0100509
9: -0.0099431, 0.0022674, -0.0097249, 0.0017128, -0.0116545, 0.0119016

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135500
time: 2.03 seconds

## Relational analysis of NS_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135501
time: 2.23 seconds

## BFS NS instance: NS_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9897457, 1.0074266, -0.0182381, 0.0176962
1: -0.0039844, 0.0005904, -0.0038191, 0.0005866, -0.0045445, 0.0044094
2: -0.0131826, 0.0110613, -0.0131624, 0.0101851, -0.0233677, 0.0240832
3: -0.0063077, 0.0047270, -0.0059089, 0.0047178, -0.0109616, 0.0106360
4: -0.0020236, 0.0026688, -0.0020197, 0.0024992, -0.0045228, 0.0046613
5: -0.0176208, 0.0128716, -0.0175954, 0.0117696, -0.0293904, 0.0302904
6: -0.0017261, 0.0060132, -0.0014464, 0.0060067, -0.0076880, 0.0074596
7: -0.0076036, 0.0124203, -0.0068799, 0.0124036, -0.0198913, 0.0193002
8: -0.0035628, 0.0069676, -0.0031822, 0.0069588, -0.0104606, 0.0101498
9: -0.0099431, 0.0022674, -0.0099329, 0.0018261, -0.0117692, 0.0121296

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135728
time: 2.20 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135728
time: 2.33 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9892271, 1.0099709, -0.0209034, 0.0186107
1: -0.0039880, 0.0006661, -0.0039483, 0.0008849, -0.0048729, 0.0046144
2: -0.0136325, 0.0110805, -0.0154553, 0.0108699, -0.0245024, 0.0265358
3: -0.0063165, 0.0049097, -0.0062206, 0.0054375, -0.0117540, 0.0111303
4: -0.0021013, 0.0026725, -0.0023257, 0.0026317, -0.0047330, 0.0049982
5: -0.0181255, 0.0128958, -0.0195840, 0.0126309, -0.0291581, 0.0305701
6: -0.0017323, 0.0062368, -0.0016650, 0.0079101, -0.0096424, 0.0079018
7: -0.0076195, 0.0127518, -0.0074455, 0.0137095, -0.0213290, 0.0201973
8: -0.0035712, 0.0071419, -0.0034797, 0.0076456, -0.0112167, 0.0106216
9: -0.0101452, 0.0022771, -0.0107293, 0.0021710, -0.0123162, 0.0130063

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0136929
time: 2.33 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0137193
time: 2.22 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1

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

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134333
time: 2.21 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134362
time: 2.22 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9886416, 1.0068365, -0.0169945, 0.0199106
1: -0.0037951, 0.0007394, -0.0040942, 0.0004395, -0.0042346, 0.0046953
2: -0.0142430, 0.0100580, -0.0123832, 0.0116430, -0.0258860, 0.0224412
3: -0.0058511, 0.0050865, -0.0065725, 0.0043632, -0.0102142, 0.0113255
4: -0.0021764, 0.0024746, -0.0018689, 0.0027814, -0.0048160, 0.0043434
5: -0.0186140, 0.0116097, -0.0166154, 0.0136033, -0.0312958, 0.0275745
6: -0.0014058, 0.0067972, -0.0019118, 0.0057580, -0.0071638, 0.0087091
7: -0.0067750, 0.0130725, -0.0080841, 0.0117600, -0.0185350, 0.0205515
8: -0.0031270, 0.0073106, -0.0038155, 0.0066204, -0.0097474, 0.0108078
9: -0.0103408, 0.0017621, -0.0095405, 0.0025604, -0.0125322, 0.0113026

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134333
time: 2.46 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134361
time: 3.00 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9900842, 1.0080066, 0.9893703, 1.0087699, -0.0186856, 0.0186362
1: -0.0037347, 0.0006834, -0.0039126, 0.0007617, -0.0044964, 0.0045960
2: -0.0137768, 0.0097380, -0.0144290, 0.0106807, -0.0244574, 0.0241670
3: -0.0057054, 0.0049515, -0.0061345, 0.0051403, -0.0108458, 0.0110860
4: -0.0021190, 0.0024127, -0.0021993, 0.0025951, -0.0047141, 0.0046120
5: -0.0182410, 0.0112073, -0.0187629, 0.0123929, -0.0299462, 0.0283237
6: -0.0013037, 0.0063692, -0.0016046, 0.0069680, -0.0082717, 0.0079738
7: -0.0065107, 0.0128275, -0.0072893, 0.0131703, -0.0196810, 0.0201168
8: -0.0029880, 0.0071817, -0.0033975, 0.0073620, -0.0103500, 0.0105792
9: -0.0101914, 0.0016009, -0.0104004, 0.0020757, -0.0122672, 0.0120014

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0125735
time: 2.17 seconds

## Relational analysis of NS_A2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0132518
time: 2.45 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9900842, 1.0080066, 0.9885285, 1.0083245, -0.0182403, 0.0194781
1: -0.0037347, 0.0006834, -0.0041224, 0.0007160, -0.0044507, 0.0048058
2: -0.0137768, 0.0097380, -0.0140485, 0.0117923, -0.0255691, 0.0237865
3: -0.0057054, 0.0049515, -0.0066405, 0.0050301, -0.0107356, 0.0115919
4: -0.0021190, 0.0024127, -0.0021525, 0.0028103, -0.0049293, 0.0045651
5: -0.0182410, 0.0112073, -0.0184584, 0.0137911, -0.0313719, 0.0281954
6: -0.0013037, 0.0063692, -0.0019595, 0.0066187, -0.0079224, 0.0083287
7: -0.0065107, 0.0128275, -0.0082074, 0.0129703, -0.0194810, 0.0210350
8: -0.0029880, 0.0071817, -0.0038804, 0.0072568, -0.0102449, 0.0110621
9: -0.0101914, 0.0016009, -0.0102785, 0.0026356, -0.0128270, 0.0118794

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0125735
time: 2.22 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0132518
time: 2.31 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9897384, 1.0098001, 0.9893038, 1.0094600, -0.0197216, 0.0204962
1: -0.0038209, 0.0008674, -0.0039292, 0.0008325, -0.0046534, 0.0047965
2: -0.0153093, 0.0101947, -0.0150187, 0.0107685, -0.0260778, 0.0252133
3: -0.0059133, 0.0053952, -0.0061745, 0.0053111, -0.0112244, 0.0115697
4: -0.0023077, 0.0025010, -0.0022719, 0.0026121, -0.0049198, 0.0047730
5: -0.0194672, 0.0117816, -0.0192347, 0.0125034, -0.0306962, 0.0293967
6: -0.0014495, 0.0077761, -0.0016327, 0.0075093, -0.0089588, 0.0094088
7: -0.0068878, 0.0136328, -0.0073618, 0.0134801, -0.0203680, 0.0209946
8: -0.0031864, 0.0076052, -0.0034357, 0.0075249, -0.0107113, 0.0110409
9: -0.0106825, 0.0018309, -0.0105894, 0.0021200, -0.0128024, 0.0124203

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0127340
time: 2.25 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0136379
time: 2.21 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9897384, 1.0098001, 0.9884640, 1.0089737, -0.0192353, 0.0213361
1: -0.0038209, 0.0008674, -0.0041384, 0.0007826, -0.0046035, 0.0050058
2: -0.0153093, 0.0101947, -0.0146033, 0.0118775, -0.0271868, 0.0247979
3: -0.0059133, 0.0053952, -0.0066793, 0.0051908, -0.0111041, 0.0120745
4: -0.0023077, 0.0025010, -0.0022208, 0.0028268, -0.0051345, 0.0047218
5: -0.0194672, 0.0117816, -0.0189023, 0.0138983, -0.0321420, 0.0292129
6: -0.0014495, 0.0077761, -0.0019867, 0.0071280, -0.0085774, 0.0097628
7: -0.0068878, 0.0136328, -0.0082778, 0.0132618, -0.0201497, 0.0219106
8: -0.0031864, 0.0076052, -0.0039174, 0.0074101, -0.0105965, 0.0115226
9: -0.0106825, 0.0018309, -0.0104563, 0.0026785, -0.0133610, 0.0122872

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0127340
time: 2.14 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0136379
time: 2.28 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9899159, 1.0071138, -0.0180550, 0.0178393
1: -0.0039902, 0.0006576, -0.0037766, 0.0005086, -0.0044988, 0.0044342
2: -0.0135619, 0.0110920, -0.0127494, 0.0099602, -0.0235221, 0.0238414
3: -0.0063217, 0.0048892, -0.0058066, 0.0045299, -0.0108516, 0.0106958
4: -0.0020926, 0.0026747, -0.0019397, 0.0024557, -0.0045482, 0.0046145
5: -0.0180690, 0.0129103, -0.0170760, 0.0114867, -0.0295557, 0.0291530
6: -0.0017359, 0.0061720, -0.0013746, 0.0058749, -0.0076108, 0.0075466
7: -0.0076290, 0.0127146, -0.0066942, 0.0120625, -0.0196915, 0.0194088
8: -0.0035762, 0.0071224, -0.0030846, 0.0067794, -0.0103556, 0.0102069
9: -0.0101226, 0.0022829, -0.0097249, 0.0017128, -0.0118354, 0.0120078

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135501
time: 2.20 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135501
time: 2.14 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9897457, 1.0074266, -0.0183678, 0.0180095
1: -0.0039902, 0.0006576, -0.0038191, 0.0005866, -0.0045768, 0.0044492
2: -0.0135619, 0.0110920, -0.0131624, 0.0101851, -0.0237470, 0.0242544
3: -0.0063217, 0.0048892, -0.0059089, 0.0047178, -0.0110396, 0.0107318
4: -0.0020926, 0.0026747, -0.0020197, 0.0024992, -0.0045635, 0.0046944
5: -0.0180690, 0.0129103, -0.0175954, 0.0117696, -0.0296552, 0.0296310
6: -0.0017359, 0.0061720, -0.0014464, 0.0060067, -0.0077427, 0.0076184
7: -0.0076290, 0.0127146, -0.0068799, 0.0124036, -0.0200327, 0.0194741
8: -0.0035762, 0.0071224, -0.0031822, 0.0069588, -0.0105350, 0.0102413
9: -0.0101226, 0.0022829, -0.0099329, 0.0018261, -0.0118752, 0.0122158

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135536
time: 1.97 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135536
time: 2.09 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9892098, 1.0099934, -0.0209622, 0.0192888
1: -0.0039971, 0.0007339, -0.0039526, 0.0008872, -0.0048843, 0.0046865
2: -0.0141972, 0.0111284, -0.0154746, 0.0108926, -0.0250898, 0.0266029
3: -0.0063383, 0.0050732, -0.0062310, 0.0054431, -0.0117814, 0.0113042
4: -0.0021708, 0.0026818, -0.0023281, 0.0026361, -0.0048069, 0.0050098
5: -0.0185774, 0.0129560, -0.0195995, 0.0126595, -0.0298428, 0.0310891
6: -0.0017475, 0.0067552, -0.0016723, 0.0079279, -0.0096754, 0.0084275
7: -0.0076590, 0.0130485, -0.0074643, 0.0137197, -0.0213787, 0.0205128
8: -0.0035920, 0.0072979, -0.0034896, 0.0076509, -0.0112429, 0.0107875
9: -0.0103261, 0.0023012, -0.0107354, 0.0021825, -0.0125086, 0.0130366

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0136929
time: 2.17 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0136979
time: 2.30 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.31 seconds
NS_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134333
NS_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
NS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134333
NS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0134956
NS_A1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0125735
NS_A1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0132519
NS_A1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0125735
NS_A1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126332, upper bound: 0.0132518
NS_A1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0127340
NS_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0136378
NS_A1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0127340
NS_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126362, upper bound: 0.0136378
NS_A1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135500
NS_A1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135501
NS_A1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135728
NS_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126347, upper bound: 0.0135728
NS_A1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0136929
NS_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0137193
NS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134333
NS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134362
NS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134333
NS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0134361
NS_A2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0125735
NS_A2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0132518
NS_A2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0125735
NS_A2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0126958, upper bound: 0.0132518
NS_A2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0127340
NS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0136379
NS_A2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0127340
NS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0136379
NS_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135501
NS_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135501
NS_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135536
NS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127013, upper bound: 0.0135536
NS_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0136929
NS_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.31
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0136979

## BFS NS instance: NS_A1_A1_B1_A2_B1_B1

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

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124318, upper bound: 0.0129923
time: 2.16 seconds

## Relational analysis of NS_A1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125279, upper bound: 0.0133061
time: 1.89 seconds

## BFS NS instance: NS_A1_A1_B1_A2_B1_B2

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

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124318, upper bound: 0.0130327
time: 2.27 seconds

## Relational analysis of NS_A1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125279, upper bound: 0.0133496
time: 1.94 seconds

## BFS NS instance: NS_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9891081, 1.0067297, -0.0166120, 0.0177421
1: -0.0037574, 0.0006411, -0.0039779, 0.0004129, -0.0041393, 0.0044208
2: -0.0134516, 0.0098585, -0.0122422, 0.0110270, -0.0234282, 0.0219360
3: -0.0057603, 0.0048494, -0.0062921, 0.0042990, -0.0099843, 0.0106635
4: -0.0020756, 0.0024360, -0.0018416, 0.0026621, -0.0045345, 0.0042457
5: -0.0179591, 0.0113588, -0.0164380, 0.0128285, -0.0294665, 0.0275897
6: -0.0013422, 0.0060990, -0.0017152, 0.0057130, -0.0070026, 0.0074789
7: -0.0066102, 0.0126424, -0.0075753, 0.0116436, -0.0181177, 0.0193502
8: -0.0030404, 0.0070844, -0.0035479, 0.0065591, -0.0095279, 0.0101761
9: -0.0100785, 0.0016616, -0.0094695, 0.0022502, -0.0117997, 0.0110481

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123374, upper bound: 0.0128843
time: 2.56 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125089, upper bound: 0.0132439
time: 2.09 seconds

## BFS NS instance: NS_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9899929, 1.0076456, 0.9890829, 1.0069813, -0.0168836, 0.0179156
1: -0.0037574, 0.0006411, -0.0039842, 0.0004756, -0.0042069, 0.0044641
2: -0.0134516, 0.0098585, -0.0125743, 0.0110603, -0.0236574, 0.0222946
3: -0.0057603, 0.0048494, -0.0063073, 0.0044502, -0.0101475, 0.0107678
4: -0.0020756, 0.0024360, -0.0019058, 0.0026686, -0.0045788, 0.0043151
5: -0.0179591, 0.0113588, -0.0168557, 0.0128704, -0.0297548, 0.0280407
6: -0.0013422, 0.0060990, -0.0017258, 0.0058190, -0.0071170, 0.0075521
7: -0.0066102, 0.0126424, -0.0076028, 0.0119179, -0.0184139, 0.0195395
8: -0.0030404, 0.0070844, -0.0035624, 0.0067034, -0.0096837, 0.0102756
9: -0.0100785, 0.0016616, -0.0096367, 0.0022669, -0.0119151, 0.0112287

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120753, upper bound: 0.0131771
time: 1.99 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125089, upper bound: 0.0133079
time: 2.18 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9899768, 1.0084147, 0.9893202, 1.0094391, -0.0194623, 0.0190945
1: -0.0037615, 0.0007253, -0.0039251, 0.0008303, -0.0045918, 0.0046503
2: -0.0141255, 0.0098799, -0.0150009, 0.0107468, -0.0248723, 0.0248807
3: -0.0057700, 0.0050524, -0.0061646, 0.0053059, -0.0110759, 0.0112170
4: -0.0021620, 0.0024401, -0.0022697, 0.0026079, -0.0047699, 0.0047099
5: -0.0185200, 0.0113857, -0.0192204, 0.0124761, -0.0287939, 0.0286125
6: -0.0013490, 0.0066894, -0.0016257, 0.0074930, -0.0088420, 0.0083151
7: -0.0066279, 0.0130108, -0.0073439, 0.0134708, -0.0200986, 0.0203547
8: -0.0030497, 0.0072781, -0.0034262, 0.0075200, -0.0105697, 0.0107043
9: -0.0103032, 0.0016724, -0.0105837, 0.0021090, -0.0124122, 0.0122560

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0136196
time: 2.27 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0136724
time: 2.45 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9899768, 1.0084147, 0.9884507, 1.0089504, -0.0189735, 0.0199640
1: -0.0037615, 0.0007253, -0.0041417, 0.0007802, -0.0045417, 0.0048670
2: -0.0141255, 0.0098799, -0.0145832, 0.0118950, -0.0260205, 0.0244630
3: -0.0057700, 0.0050524, -0.0066872, 0.0051850, -0.0109550, 0.0117397
4: -0.0021620, 0.0024401, -0.0022183, 0.0028301, -0.0049921, 0.0046584
5: -0.0185200, 0.0113857, -0.0188862, 0.0139202, -0.0302841, 0.0284267
6: -0.0013490, 0.0066894, -0.0019923, 0.0071095, -0.0084585, 0.0086817
7: -0.0066279, 0.0130108, -0.0082922, 0.0132513, -0.0198791, 0.0213030
8: -0.0030497, 0.0072781, -0.0039249, 0.0074046, -0.0104542, 0.0112031
9: -0.0103032, 0.0016724, -0.0104498, 0.0026873, -0.0129905, 0.0121222

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135758
time: 2.16 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0136378
time: 2.18 seconds

## BFS NS instance: NS_A1_A2_B1_A2_B1_B1

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

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0129960
time: 2.06 seconds

## Relational analysis of NS_A1_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124500, upper bound: 0.0133587
time: 2.08 seconds

## BFS NS instance: NS_A1_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9891081, 1.0067297, -0.0172917, 0.0174826
1: -0.0039844, 0.0005904, -0.0039779, 0.0004129, -0.0043086, 0.0043562
2: -0.0131826, 0.0110613, -0.0122422, 0.0110270, -0.0230855, 0.0228335
3: -0.0063077, 0.0047270, -0.0062921, 0.0042990, -0.0103928, 0.0105075
4: -0.0020236, 0.0026688, -0.0018416, 0.0026621, -0.0044682, 0.0044194
5: -0.0176208, 0.0128716, -0.0164380, 0.0128285, -0.0290355, 0.0287185
6: -0.0017261, 0.0060132, -0.0017152, 0.0057130, -0.0072891, 0.0073695
7: -0.0076036, 0.0124203, -0.0075753, 0.0116436, -0.0188590, 0.0190672
8: -0.0035628, 0.0069676, -0.0035479, 0.0065591, -0.0099178, 0.0100273
9: -0.0099431, 0.0022674, -0.0094695, 0.0022502, -0.0116271, 0.0115001

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0129960
time: 2.28 seconds

## Relational analysis of NS_A1_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124501, upper bound: 0.0133587
time: 2.35 seconds

## BFS NS instance: NS_A1_A2_B1_A2_B2_B1

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

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0130067
time: 2.48 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124500, upper bound: 0.0133807
time: 2.29 seconds

## BFS NS instance: NS_A1_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890820, 1.0074419, 0.9890829, 1.0069813, -0.0178213, 0.0178676
1: -0.0039844, 0.0005904, -0.0039842, 0.0004756, -0.0044406, 0.0044521
2: -0.0131826, 0.0110613, -0.0125743, 0.0110603, -0.0235939, 0.0235328
3: -0.0063077, 0.0047270, -0.0063073, 0.0044502, -0.0107111, 0.0107389
4: -0.0020236, 0.0026688, -0.0019058, 0.0026686, -0.0045666, 0.0045547
5: -0.0176208, 0.0128716, -0.0168557, 0.0128704, -0.0296749, 0.0295981
6: -0.0017261, 0.0060132, -0.0017258, 0.0058190, -0.0075123, 0.0075318
7: -0.0076036, 0.0124203, -0.0076028, 0.0119179, -0.0194366, 0.0194871
8: -0.0035628, 0.0069676, -0.0035624, 0.0067034, -0.0102215, 0.0102481
9: -0.0099431, 0.0022674, -0.0096367, 0.0022669, -0.0118831, 0.0118524

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0130068
time: 2.48 seconds

## Relational analysis of NS_A1_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124501, upper bound: 0.0133807
time: 2.26 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9897972, 1.0097408, -0.0206733, 0.0180405
1: -0.0039880, 0.0006661, -0.0038062, 0.0008613, -0.0048493, 0.0044723
2: -0.0136325, 0.0110805, -0.0152587, 0.0101169, -0.0237495, 0.0263392
3: -0.0063165, 0.0049097, -0.0058779, 0.0053806, -0.0116970, 0.0107876
4: -0.0021013, 0.0026725, -0.0023015, 0.0024860, -0.0045872, 0.0049740
5: -0.0181255, 0.0128958, -0.0194267, 0.0116839, -0.0282177, 0.0304260
6: -0.0017323, 0.0062368, -0.0014247, 0.0077297, -0.0094619, 0.0076615
7: -0.0076195, 0.0127518, -0.0068237, 0.0136062, -0.0212257, 0.0195754
8: -0.0035712, 0.0071419, -0.0031526, 0.0075912, -0.0111624, 0.0102945
9: -0.0101452, 0.0022771, -0.0106663, 0.0017918, -0.0119370, 0.0129433

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136928
time: 2.49 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136929
time: 2.31 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9896201, 1.0104679, -0.0214004, 0.0182176
1: -0.0039880, 0.0006661, -0.0038503, 0.0009359, -0.0049239, 0.0045164
2: -0.0136325, 0.0110805, -0.0158800, 0.0103508, -0.0239834, 0.0269605
3: -0.0063165, 0.0049097, -0.0059844, 0.0055605, -0.0118770, 0.0108941
4: -0.0021013, 0.0026725, -0.0023780, 0.0025313, -0.0046325, 0.0050505
5: -0.0181255, 0.0128958, -0.0199239, 0.0119781, -0.0286084, 0.0309442
6: -0.0017323, 0.0062368, -0.0014993, 0.0083000, -0.0100323, 0.0077361
7: -0.0076195, 0.0127518, -0.0070168, 0.0139327, -0.0215522, 0.0197686
8: -0.0035712, 0.0071419, -0.0032542, 0.0077629, -0.0113341, 0.0103961
9: -0.0101452, 0.0022771, -0.0108653, 0.0019096, -0.0120548, 0.0131424

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0137193
time: 2.26 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0137192
time: 2.26 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_B1

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

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124985, upper bound: 0.0129923
time: 2.55 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125947, upper bound: 0.0133061
time: 2.13 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_B2

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

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124986, upper bound: 0.0129923
time: 2.23 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125946, upper bound: 0.0133061
time: 2.25 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9891081, 1.0067297, -0.0168877, 0.0194441
1: -0.0037951, 0.0007394, -0.0039779, 0.0004129, -0.0042080, 0.0045621
2: -0.0142430, 0.0100580, -0.0122422, 0.0110270, -0.0252700, 0.0223002
3: -0.0058511, 0.0050865, -0.0062921, 0.0042990, -0.0101501, 0.0110043
4: -0.0021764, 0.0024746, -0.0018416, 0.0026621, -0.0046794, 0.0043162
5: -0.0186140, 0.0116097, -0.0164380, 0.0128285, -0.0304081, 0.0273422
6: -0.0014058, 0.0067972, -0.0017152, 0.0057130, -0.0071188, 0.0085124
7: -0.0067750, 0.0130725, -0.0075753, 0.0116436, -0.0184186, 0.0199686
8: -0.0031270, 0.0073106, -0.0035479, 0.0065591, -0.0096861, 0.0105013
9: -0.0103408, 0.0017621, -0.0094695, 0.0022502, -0.0121768, 0.0112315

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123895, upper bound: 0.0128840
time: 2.44 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125475, upper bound: 0.0132439
time: 2.43 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9898420, 1.0085522, 0.9890829, 1.0069813, -0.0171393, 0.0194693
1: -0.0037951, 0.0007394, -0.0039842, 0.0004756, -0.0042707, 0.0045148
2: -0.0142430, 0.0100580, -0.0125743, 0.0110603, -0.0253033, 0.0226323
3: -0.0058511, 0.0050865, -0.0063073, 0.0044502, -0.0103012, 0.0108902
4: -0.0021764, 0.0024746, -0.0019058, 0.0026686, -0.0046309, 0.0043804
5: -0.0186140, 0.0116097, -0.0168557, 0.0128704, -0.0300929, 0.0276170
6: -0.0014058, 0.0067972, -0.0017258, 0.0058190, -0.0072248, 0.0085230
7: -0.0067750, 0.0130725, -0.0076028, 0.0119179, -0.0186929, 0.0197616
8: -0.0031270, 0.0073106, -0.0035624, 0.0067034, -0.0098304, 0.0103924
9: -0.0103408, 0.0017621, -0.0096367, 0.0022669, -0.0120505, 0.0113988

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123895, upper bound: 0.0128861
time: 2.22 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125475, upper bound: 0.0132474
time: 2.34 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9898150, 1.0092871, 0.9893038, 1.0094600, -0.0196450, 0.0199833
1: -0.0038018, 0.0008148, -0.0039292, 0.0008325, -0.0046343, 0.0047439
2: -0.0148711, 0.0100935, -0.0150187, 0.0107685, -0.0256395, 0.0251122
3: -0.0058673, 0.0052683, -0.0061745, 0.0053111, -0.0111783, 0.0114428
4: -0.0022538, 0.0024815, -0.0022719, 0.0026121, -0.0048659, 0.0047534
5: -0.0191166, 0.0116544, -0.0192347, 0.0125034, -0.0295448, 0.0292712
6: -0.0014172, 0.0073738, -0.0016327, 0.0075093, -0.0089265, 0.0090065
7: -0.0068043, 0.0134026, -0.0073618, 0.0134801, -0.0202844, 0.0207644
8: -0.0031425, 0.0074841, -0.0034357, 0.0075249, -0.0106674, 0.0109198
9: -0.0105421, 0.0017800, -0.0105894, 0.0021200, -0.0126620, 0.0123694

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 54

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0136196
time: 2.19 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0136196
time: 3.46 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9898150, 1.0092871, 0.9884640, 1.0089737, -0.0191587, 0.0208231
1: -0.0038018, 0.0008148, -0.0041384, 0.0007826, -0.0045844, 0.0049532
2: -0.0148711, 0.0100935, -0.0146033, 0.0118775, -0.0267486, 0.0246968
3: -0.0058673, 0.0052683, -0.0066793, 0.0051908, -0.0110580, 0.0119476
4: -0.0022538, 0.0024815, -0.0022208, 0.0028268, -0.0050805, 0.0047022
5: -0.0191166, 0.0116544, -0.0189023, 0.0138983, -0.0309687, 0.0290875
6: -0.0014172, 0.0073738, -0.0019867, 0.0071280, -0.0085452, 0.0093605
7: -0.0068043, 0.0134026, -0.0082778, 0.0132618, -0.0200661, 0.0216803
8: -0.0031425, 0.0074841, -0.0039174, 0.0074101, -0.0105526, 0.0114015
9: -0.0105421, 0.0017800, -0.0104563, 0.0026785, -0.0132206, 0.0122363

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135759
time: 2.23 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135781
time: 2.32 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1_B1

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

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123718, upper bound: 0.0129960
time: 2.14 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133587
time: 2.14 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9891081, 1.0067297, -0.0176429, 0.0186471
1: -0.0039902, 0.0006576, -0.0039779, 0.0004129, -0.0044031, 0.0045116
2: -0.0135619, 0.0110920, -0.0122422, 0.0110270, -0.0245889, 0.0232972
3: -0.0063217, 0.0048892, -0.0062921, 0.0042990, -0.0106207, 0.0108825
4: -0.0020926, 0.0026747, -0.0018416, 0.0026621, -0.0046276, 0.0045163
5: -0.0180690, 0.0129103, -0.0164380, 0.0128285, -0.0300716, 0.0282229
6: -0.0017359, 0.0061720, -0.0017152, 0.0057130, -0.0074371, 0.0078872
7: -0.0076290, 0.0127146, -0.0075753, 0.0116436, -0.0192726, 0.0197476
8: -0.0035762, 0.0071224, -0.0035479, 0.0065591, -0.0101353, 0.0103851
9: -0.0101226, 0.0022829, -0.0094695, 0.0022502, -0.0120420, 0.0117524

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123717, upper bound: 0.0129960
time: 2.29 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133586
time: 2.53 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2_B1

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

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123717, upper bound: 0.0129898
time: 2.33 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133563
time: 2.39 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890589, 1.0077552, 0.9890829, 1.0069813, -0.0178210, 0.0186723
1: -0.0039902, 0.0006576, -0.0039842, 0.0004756, -0.0044658, 0.0044346
2: -0.0135619, 0.0110920, -0.0125743, 0.0110603, -0.0246222, 0.0235325
3: -0.0063217, 0.0048892, -0.0063073, 0.0044502, -0.0107719, 0.0106967
4: -0.0020926, 0.0026747, -0.0019058, 0.0026686, -0.0045486, 0.0045806
5: -0.0180690, 0.0129103, -0.0168557, 0.0128704, -0.0295582, 0.0285359
6: -0.0017359, 0.0061720, -0.0017258, 0.0058190, -0.0075122, 0.0078978
7: -0.0076290, 0.0127146, -0.0076028, 0.0119179, -0.0195469, 0.0194104
8: -0.0035762, 0.0071224, -0.0035624, 0.0067034, -0.0102795, 0.0102077
9: -0.0101226, 0.0022829, -0.0096367, 0.0022669, -0.0118364, 0.0119196

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123717, upper bound: 0.0129898
time: 2.48 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133563
time: 2.50 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9897972, 1.0097408, -0.0207096, 0.0187014
1: -0.0039971, 0.0007339, -0.0038062, 0.0008613, -0.0048584, 0.0045401
2: -0.0141972, 0.0111284, -0.0152587, 0.0101169, -0.0243142, 0.0263871
3: -0.0063383, 0.0050732, -0.0058779, 0.0053806, -0.0117188, 0.0109511
4: -0.0021708, 0.0026818, -0.0023015, 0.0024860, -0.0046568, 0.0049832
5: -0.0185774, 0.0129560, -0.0194267, 0.0116839, -0.0288536, 0.0307714
6: -0.0017475, 0.0067552, -0.0014247, 0.0077297, -0.0094772, 0.0081799
7: -0.0076590, 0.0130485, -0.0068237, 0.0136062, -0.0212653, 0.0198722
8: -0.0035920, 0.0072979, -0.0031526, 0.0075912, -0.0111832, 0.0104506
9: -0.0103261, 0.0023012, -0.0106663, 0.0017918, -0.0121179, 0.0129675

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136929
time: 2.10 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136928
time: 2.43 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9896201, 1.0104679, -0.0214366, 0.0188785
1: -0.0039971, 0.0007339, -0.0038503, 0.0009359, -0.0049330, 0.0045842
2: -0.0141972, 0.0111284, -0.0158800, 0.0103508, -0.0245480, 0.0270083
3: -0.0063383, 0.0050732, -0.0059844, 0.0055605, -0.0118987, 0.0110576
4: -0.0021708, 0.0026818, -0.0023780, 0.0025313, -0.0047021, 0.0050598
5: -0.0185774, 0.0129560, -0.0199239, 0.0119781, -0.0288179, 0.0312122
6: -0.0017475, 0.0067552, -0.0014993, 0.0083000, -0.0100476, 0.0082545
7: -0.0076590, 0.0130485, -0.0070168, 0.0139327, -0.0215917, 0.0200653
8: -0.0035920, 0.0072979, -0.0032542, 0.0077629, -0.0113549, 0.0105522
9: -0.0103261, 0.0023012, -0.0108653, 0.0019096, -0.0122357, 0.0131665

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136979
time: 2.22 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136979
time: 2.62 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.68 seconds
NS_A1_A1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124318, upper bound: 0.0129923
NS_A1_A1_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125279, upper bound: 0.0133061
NS_A1_A1_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124318, upper bound: 0.0130327
NS_A1_A1_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125279, upper bound: 0.0133496
NS_A1_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123374, upper bound: 0.0128843
NS_A1_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125089, upper bound: 0.0132439
NS_A1_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0120753, upper bound: 0.0131771
NS_A1_A1_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125089, upper bound: 0.0133079
NS_A1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0136196
NS_A1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0127095, upper bound: 0.0136724
NS_A1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0135758
NS_A1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0126974, upper bound: 0.0136378
NS_A1_A2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0129960
NS_A1_A2_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124500, upper bound: 0.0133587
NS_A1_A2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0129960
NS_A1_A2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124501, upper bound: 0.0133587
NS_A1_A2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0130067
NS_A1_A2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124500, upper bound: 0.0133807
NS_A1_A2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0122998, upper bound: 0.0130068
NS_A1_A2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124501, upper bound: 0.0133807
NS_A1_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136928
NS_A1_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0136929
NS_A1_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0137193
NS_A1_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129147, upper bound: 0.0137192
NS_A2_A1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124985, upper bound: 0.0129923
NS_A2_A1_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125947, upper bound: 0.0133061
NS_A2_A1_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0124986, upper bound: 0.0129923
NS_A2_A1_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125946, upper bound: 0.0133061
NS_A2_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123895, upper bound: 0.0128840
NS_A2_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125475, upper bound: 0.0132439
NS_A2_A1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123895, upper bound: 0.0128861
NS_A2_A1_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125475, upper bound: 0.0132474
NS_A2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0136196
NS_A2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0127751, upper bound: 0.0136196
NS_A2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135759
NS_A2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0127323, upper bound: 0.0135781
NS_A2_A2_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123718, upper bound: 0.0129960
NS_A2_A2_B1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133587
NS_A2_A2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123717, upper bound: 0.0129960
NS_A2_A2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133586
NS_A2_A2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123717, upper bound: 0.0129898
NS_A2_A2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133563
NS_A2_A2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0123717, upper bound: 0.0129898
NS_A2_A2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0125207, upper bound: 0.0133563
NS_A2_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136929
NS_A2_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136928
NS_A2_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136979
NS_A2_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.68
Output dim: 0, lower bound: -0.0129764, upper bound: 0.0136979

## BFS NS instance: NS_A1_A1_B2_A2_B1_A2_B1

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

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123925, upper bound: 0.0132737
time: 2.27 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127785, upper bound: 0.0134253
time: 2.18 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B1_A2_B2

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

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123925, upper bound: 0.0133268
time: 2.01 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127785, upper bound: 0.0134797
time: 2.24 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9899768, 1.0084147, 0.9889733, 1.0087116, -0.0187348, 0.0194415
1: -0.0037615, 0.0007253, -0.0040115, 0.0007557, -0.0045172, 0.0047368
2: -0.0141255, 0.0098799, -0.0143792, 0.0112051, -0.0253306, 0.0242591
3: -0.0057700, 0.0050524, -0.0063732, 0.0051259, -0.0108959, 0.0114256
4: -0.0021620, 0.0024401, -0.0021932, 0.0026966, -0.0048586, 0.0046333
5: -0.0185200, 0.0113857, -0.0187230, 0.0130525, -0.0291181, 0.0282749
6: -0.0013490, 0.0066894, -0.0017720, 0.0069223, -0.0082712, 0.0084614
7: -0.0066279, 0.0130108, -0.0077224, 0.0131441, -0.0197720, 0.0207332
8: -0.0030497, 0.0072781, -0.0036253, 0.0073482, -0.0103979, 0.0109034
9: -0.0103032, 0.0016724, -0.0103845, 0.0023398, -0.0126430, 0.0120569

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126258, upper bound: 0.0130004
time: 2.09 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128039, upper bound: 0.0133834
time: 2.48 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9899768, 1.0084147, 0.9889344, 1.0093356, -0.0193588, 0.0194803
1: -0.0037615, 0.0007253, -0.0040212, 0.0008197, -0.0045812, 0.0047465
2: -0.0141255, 0.0098799, -0.0149125, 0.0112562, -0.0253817, 0.0247924
3: -0.0057700, 0.0050524, -0.0063965, 0.0052803, -0.0110504, 0.0114489
4: -0.0021620, 0.0024401, -0.0022589, 0.0027065, -0.0048685, 0.0046990
5: -0.0185200, 0.0113857, -0.0191498, 0.0131168, -0.0295404, 0.0286765
6: -0.0013490, 0.0066894, -0.0017884, 0.0074119, -0.0087609, 0.0084777
7: -0.0066279, 0.0130108, -0.0077646, 0.0134243, -0.0200522, 0.0207754
8: -0.0030497, 0.0072781, -0.0036475, 0.0074956, -0.0105453, 0.0109256
9: -0.0103032, 0.0016724, -0.0105554, 0.0023656, -0.0126688, 0.0122277

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126258, upper bound: 0.0130514
time: 2.13 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128039, upper bound: 0.0134476
time: 2.53 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B1

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

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133225
time: 2.10 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127283, upper bound: 0.0135017
time: 2.41 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9889733, 1.0087116, -0.0196441, 0.0188645
1: -0.0039880, 0.0006661, -0.0040115, 0.0007557, -0.0047438, 0.0046776
2: -0.0136325, 0.0110805, -0.0143792, 0.0112051, -0.0248376, 0.0254597
3: -0.0063165, 0.0049097, -0.0063732, 0.0051259, -0.0114424, 0.0112829
4: -0.0021013, 0.0026725, -0.0021932, 0.0026966, -0.0047979, 0.0048657
5: -0.0181255, 0.0128958, -0.0187230, 0.0130525, -0.0284069, 0.0293817
6: -0.0017323, 0.0062368, -0.0017720, 0.0069223, -0.0086545, 0.0080088
7: -0.0076195, 0.0127518, -0.0077224, 0.0131441, -0.0207636, 0.0204741
8: -0.0035712, 0.0071419, -0.0036253, 0.0073482, -0.0109194, 0.0107672
9: -0.0101452, 0.0022771, -0.0103845, 0.0023398, -0.0124850, 0.0126615

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133224
time: 2.47 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0135016
time: 2.74 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9897113, 1.0100468, -0.0209793, 0.0181265
1: -0.0039880, 0.0006661, -0.0038276, 0.0008927, -0.0048807, 0.0044937
2: -0.0136325, 0.0110805, -0.0155202, 0.0102304, -0.0238629, 0.0266007
3: -0.0063165, 0.0049097, -0.0059296, 0.0054563, -0.0117728, 0.0108393
4: -0.0021013, 0.0026725, -0.0023337, 0.0025080, -0.0046092, 0.0050062
5: -0.0181255, 0.0128958, -0.0196360, 0.0118266, -0.0280794, 0.0306769
6: -0.0017323, 0.0062368, -0.0014609, 0.0079698, -0.0097020, 0.0076977
7: -0.0076195, 0.0127518, -0.0069174, 0.0137437, -0.0213632, 0.0196692
8: -0.0035712, 0.0071419, -0.0032019, 0.0076635, -0.0112347, 0.0103438
9: -0.0101452, 0.0022771, -0.0107501, 0.0018489, -0.0119942, 0.0130272

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133546
time: 2.97 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127283, upper bound: 0.0135271
time: 2.36 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890675, 1.0078378, 0.9889288, 1.0093356, -0.0202681, 0.0189090
1: -0.0039880, 0.0006661, -0.0040226, 0.0008197, -0.0048078, 0.0046887
2: -0.0136325, 0.0110805, -0.0149125, 0.0112636, -0.0248962, 0.0259930
3: -0.0063165, 0.0049097, -0.0063998, 0.0052803, -0.0115968, 0.0113095
4: -0.0021013, 0.0026725, -0.0022589, 0.0027079, -0.0048092, 0.0049314
5: -0.0181255, 0.0128958, -0.0191498, 0.0131261, -0.0289603, 0.0299904
6: -0.0017323, 0.0062368, -0.0017907, 0.0074119, -0.0091441, 0.0080275
7: -0.0076195, 0.0127518, -0.0077708, 0.0134243, -0.0210438, 0.0205225
8: -0.0035712, 0.0071419, -0.0036507, 0.0074956, -0.0110668, 0.0107926
9: -0.0101452, 0.0022771, -0.0105554, 0.0023693, -0.0125145, 0.0128324

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133546
time: 2.44 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127283, upper bound: 0.0135271
time: 2.66 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9898150, 1.0092871, 0.9898834, 1.0092132, -0.0193982, 0.0194037
1: -0.0038018, 0.0008148, -0.0037847, 0.0008072, -0.0046090, 0.0045995
2: -0.0148711, 0.0100935, -0.0148080, 0.0100032, -0.0248742, 0.0249015
3: -0.0058673, 0.0052683, -0.0058261, 0.0052501, -0.0111173, 0.0110945
4: -0.0022538, 0.0024815, -0.0022460, 0.0024640, -0.0047177, 0.0047275
5: -0.0191166, 0.0116544, -0.0190661, 0.0115408, -0.0282686, 0.0291241
6: -0.0014172, 0.0073738, -0.0013883, 0.0073159, -0.0087331, 0.0087622
7: -0.0068043, 0.0134026, -0.0067297, 0.0133694, -0.0201737, 0.0201322
8: -0.0031425, 0.0074841, -0.0031032, 0.0074667, -0.0106092, 0.0105873
9: -0.0105421, 0.0017800, -0.0105219, 0.0017345, -0.0122766, 0.0123018

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124639, upper bound: 0.0132737
time: 2.23 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128438, upper bound: 0.0134253
time: 2.13 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898150, 1.0092871, 0.9897113, 1.0100468, -0.0202318, 0.0195758
1: -0.0038018, 0.0008148, -0.0038276, 0.0008927, -0.0046945, 0.0046424
2: -0.0148711, 0.0100935, -0.0155202, 0.0102304, -0.0251015, 0.0256138
3: -0.0058673, 0.0052683, -0.0059296, 0.0054563, -0.0113236, 0.0111979
4: -0.0022538, 0.0024815, -0.0023337, 0.0025080, -0.0047617, 0.0048152
5: -0.0191166, 0.0116544, -0.0196360, 0.0118266, -0.0285763, 0.0294572
6: -0.0014172, 0.0073738, -0.0014609, 0.0079698, -0.0093870, 0.0088347
7: -0.0068043, 0.0134026, -0.0069174, 0.0137437, -0.0205480, 0.0203199
8: -0.0031425, 0.0074841, -0.0032019, 0.0076635, -0.0108060, 0.0106861
9: -0.0105421, 0.0017800, -0.0107501, 0.0018489, -0.0123910, 0.0125301

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124639, upper bound: 0.0132737
time: 2.53 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128437, upper bound: 0.0134255
time: 2.12 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9898150, 1.0092871, 0.9889733, 1.0087116, -0.0188966, 0.0203139
1: -0.0038018, 0.0008148, -0.0040115, 0.0007557, -0.0045575, 0.0048263
2: -0.0148711, 0.0100935, -0.0143792, 0.0112051, -0.0260761, 0.0244727
3: -0.0058673, 0.0052683, -0.0063732, 0.0051259, -0.0109931, 0.0116415
4: -0.0022538, 0.0024815, -0.0021932, 0.0026966, -0.0049504, 0.0046747
5: -0.0191166, 0.0116544, -0.0187230, 0.0130525, -0.0298514, 0.0289332
6: -0.0014172, 0.0073738, -0.0017720, 0.0069223, -0.0083394, 0.0091458
7: -0.0068043, 0.0134026, -0.0077224, 0.0131441, -0.0199484, 0.0211249
8: -0.0031425, 0.0074841, -0.0036253, 0.0073482, -0.0104907, 0.0111094
9: -0.0105421, 0.0017800, -0.0103845, 0.0023398, -0.0128819, 0.0121645

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126643, upper bound: 0.0130005
time: 2.23 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128334, upper bound: 0.0133834
time: 2.66 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9898150, 1.0092871, 0.9889344, 1.0093356, -0.0195206, 0.0203527
1: -0.0038018, 0.0008148, -0.0040212, 0.0008197, -0.0046215, 0.0048360
2: -0.0148711, 0.0100935, -0.0149125, 0.0112562, -0.0261273, 0.0250060
3: -0.0058673, 0.0052683, -0.0063965, 0.0052803, -0.0111476, 0.0116648
4: -0.0022538, 0.0024815, -0.0022589, 0.0027065, -0.0049603, 0.0047403
5: -0.0191166, 0.0116544, -0.0191498, 0.0131168, -0.0300045, 0.0291462
6: -0.0014172, 0.0073738, -0.0017884, 0.0074119, -0.0088291, 0.0091622
7: -0.0068043, 0.0134026, -0.0077646, 0.0134243, -0.0202287, 0.0211672
8: -0.0031425, 0.0074841, -0.0036475, 0.0074956, -0.0106380, 0.0111316
9: -0.0105421, 0.0017800, -0.0105554, 0.0023656, -0.0129077, 0.0123353

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126643, upper bound: 0.0130008
time: 2.50 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128334, upper bound: 0.0133870
time: 2.24 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9898834, 1.0092132, -0.0201820, 0.0186152
1: -0.0039971, 0.0007339, -0.0037847, 0.0008072, -0.0048043, 0.0045186
2: -0.0141972, 0.0111284, -0.0148080, 0.0100032, -0.0242004, 0.0259364
3: -0.0063383, 0.0050732, -0.0058261, 0.0052501, -0.0115883, 0.0108993
4: -0.0021708, 0.0026818, -0.0022460, 0.0024640, -0.0046348, 0.0049278
5: -0.0185774, 0.0129560, -0.0190661, 0.0115408, -0.0278267, 0.0303766
6: -0.0017475, 0.0067552, -0.0013883, 0.0073159, -0.0090635, 0.0081436
7: -0.0076590, 0.0130485, -0.0067297, 0.0133694, -0.0210285, 0.0197782
8: -0.0035920, 0.0072979, -0.0031032, 0.0074667, -0.0110587, 0.0104011
9: -0.0103261, 0.0023012, -0.0105219, 0.0017345, -0.0120606, 0.0128231

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124044, upper bound: 0.0133225
time: 2.67 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127912, upper bound: 0.0135017
time: 2.47 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9889733, 1.0087116, -0.0196803, 0.0195254
1: -0.0039971, 0.0007339, -0.0040115, 0.0007557, -0.0047528, 0.0047454
2: -0.0141972, 0.0111284, -0.0143792, 0.0112051, -0.0254023, 0.0255075
3: -0.0063383, 0.0050732, -0.0063732, 0.0051259, -0.0114642, 0.0114464
4: -0.0021708, 0.0026818, -0.0021932, 0.0026966, -0.0048674, 0.0048750
5: -0.0185774, 0.0129560, -0.0187230, 0.0130525, -0.0290766, 0.0298460
6: -0.0017475, 0.0067552, -0.0017720, 0.0069223, -0.0086698, 0.0085272
7: -0.0076590, 0.0130485, -0.0077224, 0.0131441, -0.0208031, 0.0207709
8: -0.0035920, 0.0072979, -0.0036253, 0.0073482, -0.0109402, 0.0109232
9: -0.0103261, 0.0023012, -0.0103845, 0.0023398, -0.0126660, 0.0126857

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124044, upper bound: 0.0133225
time: 2.58 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127912, upper bound: 0.0135017
time: 2.49 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9897113, 1.0100468, -0.0210156, 0.0187874
1: -0.0039971, 0.0007339, -0.0038276, 0.0008927, -0.0048898, 0.0045615
2: -0.0141972, 0.0111284, -0.0155202, 0.0102304, -0.0244276, 0.0266486
3: -0.0063383, 0.0050732, -0.0059296, 0.0054563, -0.0117946, 0.0110028
4: -0.0021708, 0.0026818, -0.0023337, 0.0025080, -0.0046787, 0.0050155
5: -0.0185774, 0.0129560, -0.0196360, 0.0118266, -0.0282733, 0.0308833
6: -0.0017475, 0.0067552, -0.0014609, 0.0079698, -0.0097173, 0.0082161
7: -0.0076590, 0.0130485, -0.0069174, 0.0137437, -0.0214027, 0.0199659
8: -0.0035920, 0.0072979, -0.0032019, 0.0076635, -0.0112555, 0.0104999
9: -0.0103261, 0.0023012, -0.0107501, 0.0018489, -0.0121751, 0.0130513

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124043, upper bound: 0.0133210
time: 2.56 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127913, upper bound: 0.0134992
time: 2.40 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890313, 1.0084987, 0.9889288, 1.0093356, -0.0203044, 0.0195699
1: -0.0039971, 0.0007339, -0.0040226, 0.0008197, -0.0048168, 0.0047565
2: -0.0141972, 0.0111284, -0.0149125, 0.0112636, -0.0254608, 0.0260409
3: -0.0063383, 0.0050732, -0.0063998, 0.0052803, -0.0116186, 0.0114730
4: -0.0021708, 0.0026818, -0.0022589, 0.0027079, -0.0048787, 0.0049406
5: -0.0185774, 0.0129560, -0.0191498, 0.0131261, -0.0292264, 0.0301216
6: -0.0017475, 0.0067552, -0.0017907, 0.0074119, -0.0091594, 0.0085459
7: -0.0076590, 0.0130485, -0.0077708, 0.0134243, -0.0210834, 0.0208192
8: -0.0035920, 0.0072979, -0.0036507, 0.0074956, -0.0110876, 0.0109486
9: -0.0103261, 0.0023012, -0.0105554, 0.0023693, -0.0126955, 0.0128566

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124044, upper bound: 0.0133210
time: 2.33 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0127913, upper bound: 0.0134992
time: 2.46 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 5.61 seconds
NS_A1_A1_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0123925, upper bound: 0.0132737
NS_A1_A1_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127785, upper bound: 0.0134253
NS_A1_A1_B2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0123925, upper bound: 0.0133268
NS_A1_A1_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127785, upper bound: 0.0134797
NS_A1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0126258, upper bound: 0.0130004
NS_A1_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0128039, upper bound: 0.0133834
NS_A1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0126258, upper bound: 0.0130514
NS_A1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0128039, upper bound: 0.0134476
NS_A1_A2_B2_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133225
NS_A1_A2_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127283, upper bound: 0.0135017
NS_A1_A2_B2_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133224
NS_A1_A2_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127282, upper bound: 0.0135016
NS_A1_A2_B2_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133546
NS_A1_A2_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127283, upper bound: 0.0135271
NS_A1_A2_B2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0123347, upper bound: 0.0133546
NS_A1_A2_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127283, upper bound: 0.0135271
NS_A2_A1_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0124639, upper bound: 0.0132737
NS_A2_A1_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0128438, upper bound: 0.0134253
NS_A2_A1_B2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0124639, upper bound: 0.0132737
NS_A2_A1_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0128437, upper bound: 0.0134255
NS_A2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0126643, upper bound: 0.0130005
NS_A2_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0128334, upper bound: 0.0133834
NS_A2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0126643, upper bound: 0.0130008
NS_A2_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0128334, upper bound: 0.0133870
NS_A2_A2_B2_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0124044, upper bound: 0.0133225
NS_A2_A2_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127912, upper bound: 0.0135017
NS_A2_A2_B2_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0124044, upper bound: 0.0133225
NS_A2_A2_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127912, upper bound: 0.0135017
NS_A2_A2_B2_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0124043, upper bound: 0.0133210
NS_A2_A2_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127913, upper bound: 0.0134992
NS_A2_A2_B2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0124044, upper bound: 0.0133210
NS_A2_A2_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.61
Output dim: 0, lower bound: -0.0127913, upper bound: 0.0134992

## BFS NS instance: NS_A1_A1_B2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9899782, 1.0083975, 0.9899327, 1.0086417, -0.0186635, 0.0184648
1: -0.0037611, 0.0007235, -0.0037725, 0.0007486, -0.0045097, 0.0044960
2: -0.0141108, 0.0098779, -0.0143196, 0.0099381, -0.0240489, 0.0241975
3: -0.0057691, 0.0050482, -0.0057965, 0.0051086, -0.0108778, 0.0108447
4: -0.0021601, 0.0024397, -0.0021858, 0.0024514, -0.0046115, 0.0046256
5: -0.0185083, 0.0113833, -0.0186753, 0.0114589, -0.0274400, 0.0275682
6: -0.0013484, 0.0066759, -0.0013676, 0.0068675, -0.0082159, 0.0080435
7: -0.0066263, 0.0130031, -0.0066759, 0.0131128, -0.0197390, 0.0196790
8: -0.0030488, 0.0072740, -0.0030749, 0.0073317, -0.0103806, 0.0103490
9: -0.0102985, 0.0016714, -0.0103654, 0.0017017, -0.0120002, 0.0120368

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123918, upper bound: 0.0131004
time: 2.35 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124504, upper bound: 0.0130975
time: 2.51 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9899782, 1.0083975, 0.9897752, 1.0094602, -0.0194820, 0.0186222
1: -0.0037611, 0.0007235, -0.0038117, 0.0008325, -0.0045936, 0.0045352
2: -0.0141108, 0.0098779, -0.0150188, 0.0101460, -0.0242568, 0.0248968
3: -0.0057691, 0.0050482, -0.0058911, 0.0053111, -0.0110802, 0.0109393
4: -0.0021601, 0.0024397, -0.0022719, 0.0024916, -0.0046518, 0.0047117
5: -0.0185083, 0.0113833, -0.0192348, 0.0117204, -0.0281365, 0.0282173
6: -0.0013484, 0.0066759, -0.0014339, 0.0075095, -0.0088578, 0.0081098
7: -0.0066263, 0.0130031, -0.0068476, 0.0134802, -0.0201065, 0.0198507
8: -0.0030488, 0.0072740, -0.0031652, 0.0075250, -0.0105738, 0.0104393
9: -0.0102985, 0.0016714, -0.0105894, 0.0018064, -0.0121049, 0.0122608

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124505, upper bound: 0.0130718
time: 1.92 seconds

## Relational analysis of NS_A1_A1_B2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124504, upper bound: 0.0131532
time: 2.61 seconds

## BFS NS instance: NS_A1_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9900260, 1.0078346, 0.9889364, 1.0093179, -0.0192919, 0.0188981
1: -0.0037492, 0.0006658, -0.0040207, 0.0008179, -0.0045671, 0.0046865
2: -0.0136298, 0.0098148, -0.0148972, 0.0112537, -0.0248835, 0.0247120
3: -0.0057404, 0.0049089, -0.0063953, 0.0052759, -0.0110163, 0.0113042
4: -0.0021009, 0.0024275, -0.0022570, 0.0027060, -0.0048069, 0.0046845
5: -0.0181233, 0.0113039, -0.0191375, 0.0131137, -0.0286970, 0.0285826
6: -0.0013282, 0.0062343, -0.0017876, 0.0073978, -0.0087261, 0.0080218
7: -0.0065741, 0.0127503, -0.0077626, 0.0134163, -0.0199904, 0.0205129
8: -0.0030214, 0.0071411, -0.0036464, 0.0074914, -0.0105128, 0.0107875
9: -0.0101443, 0.0016396, -0.0105504, 0.0023643, -0.0125087, 0.0121901

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125726, upper bound: 0.0133895
time: 2.28 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125725, upper bound: 0.0134476
time: 2.53 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9899327, 1.0086417, -0.0195727, 0.0178868
1: -0.0039877, 0.0006642, -0.0037725, 0.0007486, -0.0047362, 0.0044367
2: -0.0136169, 0.0110785, -0.0143196, 0.0099381, -0.0235549, 0.0253981
3: -0.0063156, 0.0049052, -0.0057965, 0.0051086, -0.0114242, 0.0107017
4: -0.0020993, 0.0026721, -0.0021858, 0.0024514, -0.0045507, 0.0048580
5: -0.0181130, 0.0128933, -0.0186753, 0.0114589, -0.0272730, 0.0291807
6: -0.0017316, 0.0062224, -0.0013676, 0.0068675, -0.0085992, 0.0075900
7: -0.0076179, 0.0127435, -0.0066759, 0.0131128, -0.0207306, 0.0194195
8: -0.0035703, 0.0071376, -0.0030749, 0.0073317, -0.0109021, 0.0102125
9: -0.0101402, 0.0022761, -0.0103654, 0.0017017, -0.0118419, 0.0126415

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124587, upper bound: 0.0134591
time: 2.41 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124587, upper bound: 0.0135017
time: 2.68 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9890235, 1.0081004, -0.0190313, 0.0187959
1: -0.0039877, 0.0006642, -0.0039990, 0.0006930, -0.0046807, 0.0046632
2: -0.0136169, 0.0110785, -0.0138569, 0.0111387, -0.0247555, 0.0249355
3: -0.0063156, 0.0049052, -0.0063429, 0.0049747, -0.0112903, 0.0112481
4: -0.0020993, 0.0026721, -0.0021289, 0.0026838, -0.0047831, 0.0048010
5: -0.0181130, 0.0128933, -0.0183051, 0.0129689, -0.0283103, 0.0284754
6: -0.0017316, 0.0062224, -0.0017508, 0.0064428, -0.0081745, 0.0079733
7: -0.0076179, 0.0127435, -0.0076675, 0.0128697, -0.0204876, 0.0204111
8: -0.0035703, 0.0071376, -0.0035964, 0.0072039, -0.0107742, 0.0107340
9: -0.0101402, 0.0022761, -0.0102171, 0.0023064, -0.0124466, 0.0124932

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0134592
time: 2.58 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0135017
time: 2.35 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9897752, 1.0094602, -0.0203912, 0.0180442
1: -0.0039877, 0.0006642, -0.0038117, 0.0008325, -0.0048202, 0.0044759
2: -0.0136169, 0.0110785, -0.0150188, 0.0101460, -0.0237628, 0.0260974
3: -0.0063156, 0.0049052, -0.0058911, 0.0053111, -0.0116267, 0.0107963
4: -0.0020993, 0.0026721, -0.0022719, 0.0024916, -0.0045909, 0.0049441
5: -0.0181130, 0.0128933, -0.0192348, 0.0117204, -0.0279665, 0.0298297
6: -0.0017316, 0.0062224, -0.0014339, 0.0075095, -0.0092411, 0.0076564
7: -0.0076179, 0.0127435, -0.0068476, 0.0134802, -0.0210981, 0.0195912
8: -0.0035703, 0.0071376, -0.0031652, 0.0075250, -0.0110953, 0.0103028
9: -0.0101402, 0.0022761, -0.0105894, 0.0018064, -0.0119466, 0.0128655

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123405, upper bound: 0.0131746
time: 2.35 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123866, upper bound: 0.0131586
time: 2.14 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9889930, 1.0087382, -0.0196691, 0.0188264
1: -0.0039877, 0.0006642, -0.0040066, 0.0007584, -0.0047461, 0.0046708
2: -0.0136169, 0.0110785, -0.0144019, 0.0111787, -0.0247956, 0.0254804
3: -0.0063156, 0.0049052, -0.0063612, 0.0051325, -0.0114481, 0.0112663
4: -0.0020993, 0.0026721, -0.0021960, 0.0026915, -0.0047908, 0.0048681
5: -0.0181130, 0.0128933, -0.0187412, 0.0130193, -0.0288453, 0.0290778
6: -0.0017316, 0.0062224, -0.0017636, 0.0069431, -0.0086747, 0.0079860
7: -0.0076179, 0.0127435, -0.0077006, 0.0131560, -0.0207739, 0.0204442
8: -0.0035703, 0.0071376, -0.0036138, 0.0073545, -0.0109248, 0.0107514
9: -0.0101402, 0.0022761, -0.0103917, 0.0023266, -0.0124667, 0.0126679

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0134732
time: 2.68 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0135271
time: 2.34 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9898169, 1.0092694, 0.9899327, 1.0086417, -0.0188248, 0.0193367
1: -0.0038013, 0.0008129, -0.0037725, 0.0007486, -0.0045499, 0.0045854
2: -0.0148558, 0.0100910, -0.0143196, 0.0099381, -0.0247939, 0.0244106
3: -0.0058661, 0.0052639, -0.0057965, 0.0051086, -0.0109747, 0.0110604
4: -0.0022519, 0.0024810, -0.0021858, 0.0024514, -0.0047033, 0.0046668
5: -0.0191044, 0.0116513, -0.0186753, 0.0114589, -0.0281730, 0.0283027
6: -0.0014164, 0.0073598, -0.0013676, 0.0068675, -0.0082839, 0.0087274
7: -0.0068023, 0.0133946, -0.0066759, 0.0131128, -0.0199150, 0.0200705
8: -0.0031414, 0.0074799, -0.0030749, 0.0073317, -0.0104731, 0.0105549
9: -0.0105372, 0.0017787, -0.0103654, 0.0017017, -0.0122389, 0.0121441

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1_B2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124749, upper bound: 0.0131004
time: 2.26 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1_B2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125101, upper bound: 0.0130975
time: 2.17 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9898169, 1.0092694, 0.9897752, 1.0094602, -0.0196433, 0.0194941
1: -0.0038013, 0.0008129, -0.0038117, 0.0008325, -0.0046338, 0.0046246
2: -0.0148558, 0.0100910, -0.0150188, 0.0101460, -0.0250018, 0.0251098
3: -0.0058661, 0.0052639, -0.0058911, 0.0053111, -0.0111772, 0.0111550
4: -0.0022519, 0.0024810, -0.0022719, 0.0024916, -0.0047435, 0.0047529
5: -0.0191044, 0.0116513, -0.0192348, 0.0117204, -0.0284657, 0.0286316
6: -0.0014164, 0.0073598, -0.0014339, 0.0075095, -0.0089259, 0.0087938
7: -0.0068023, 0.0133946, -0.0068476, 0.0134802, -0.0202824, 0.0202422
8: -0.0031414, 0.0074799, -0.0031652, 0.0075250, -0.0106663, 0.0106452
9: -0.0105372, 0.0017787, -0.0105894, 0.0018064, -0.0123436, 0.0123681

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2_B2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124749, upper bound: 0.0130978
time: 2.29 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2_B2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125101, upper bound: 0.0130953
time: 2.16 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9899327, 1.0086417, -0.0196085, 0.0185478
1: -0.0039966, 0.0007320, -0.0037725, 0.0007486, -0.0047451, 0.0045045
2: -0.0141817, 0.0111258, -0.0143196, 0.0099381, -0.0241198, 0.0254454
3: -0.0063371, 0.0050687, -0.0057965, 0.0051086, -0.0114458, 0.0108652
4: -0.0021689, 0.0026813, -0.0021858, 0.0024514, -0.0046203, 0.0048671
5: -0.0185650, 0.0129528, -0.0186753, 0.0114589, -0.0277303, 0.0295832
6: -0.0017467, 0.0067410, -0.0013676, 0.0068675, -0.0086143, 0.0081086
7: -0.0076569, 0.0130403, -0.0066759, 0.0131128, -0.0207697, 0.0197163
8: -0.0035908, 0.0072936, -0.0030749, 0.0073317, -0.0109226, 0.0103686
9: -0.0103212, 0.0022999, -0.0103654, 0.0017017, -0.0120229, 0.0126653

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0134591
time: 2.55 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0135017
time: 2.70 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9890235, 1.0081004, -0.0190672, 0.0194569
1: -0.0039966, 0.0007320, -0.0039990, 0.0006930, -0.0046896, 0.0047310
2: -0.0141817, 0.0111258, -0.0138569, 0.0111387, -0.0253204, 0.0249828
3: -0.0063371, 0.0050687, -0.0063429, 0.0049747, -0.0113118, 0.0114117
4: -0.0021689, 0.0026813, -0.0021289, 0.0026838, -0.0048526, 0.0048101
5: -0.0185650, 0.0129528, -0.0183051, 0.0129689, -0.0289800, 0.0289831
6: -0.0017467, 0.0067410, -0.0017508, 0.0064428, -0.0081896, 0.0084918
7: -0.0076569, 0.0130403, -0.0076675, 0.0128697, -0.0205266, 0.0207079
8: -0.0035908, 0.0072936, -0.0035964, 0.0072039, -0.0107947, 0.0108901
9: -0.0103212, 0.0022999, -0.0102171, 0.0023064, -0.0126276, 0.0125170

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125600, upper bound: 0.0134591
time: 2.43 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125600, upper bound: 0.0135017
time: 2.51 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9897752, 1.0094602, -0.0204270, 0.0187052
1: -0.0039966, 0.0007320, -0.0038117, 0.0008325, -0.0048291, 0.0045437
2: -0.0141817, 0.0111258, -0.0150188, 0.0101460, -0.0243277, 0.0261446
3: -0.0063371, 0.0050687, -0.0058911, 0.0053111, -0.0116482, 0.0109598
4: -0.0021689, 0.0026813, -0.0022719, 0.0024916, -0.0046605, 0.0049532
5: -0.0185650, 0.0129528, -0.0192348, 0.0117204, -0.0281619, 0.0301045
6: -0.0017467, 0.0067410, -0.0014339, 0.0075095, -0.0092562, 0.0081749
7: -0.0076569, 0.0130403, -0.0068476, 0.0134802, -0.0211371, 0.0198880
8: -0.0035908, 0.0072936, -0.0031652, 0.0075250, -0.0111158, 0.0104589
9: -0.0103212, 0.0022999, -0.0105894, 0.0018064, -0.0121276, 0.0128893

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0134507
time: 2.38 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0134992
time: 2.55 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9889930, 1.0087382, -0.0197049, 0.0194874
1: -0.0039966, 0.0007320, -0.0040066, 0.0007584, -0.0047550, 0.0047386
2: -0.0141817, 0.0111258, -0.0144019, 0.0111787, -0.0253605, 0.0255277
3: -0.0063371, 0.0050687, -0.0063612, 0.0051325, -0.0114696, 0.0114299
4: -0.0021689, 0.0026813, -0.0021960, 0.0026915, -0.0048604, 0.0048772
5: -0.0185650, 0.0129528, -0.0187412, 0.0130193, -0.0291132, 0.0293006
6: -0.0017467, 0.0067410, -0.0017636, 0.0069431, -0.0086898, 0.0085046
7: -0.0076569, 0.0130403, -0.0077006, 0.0131560, -0.0208130, 0.0207410
8: -0.0035908, 0.0072936, -0.0036138, 0.0073545, -0.0109453, 0.0109075
9: -0.0103212, 0.0022999, -0.0103917, 0.0023266, -0.0126477, 0.0126916

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125601, upper bound: 0.0134507
time: 2.04 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125601, upper bound: 0.0134992
time: 2.76 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 5.64 seconds
NS_A1_A1_B2_A2_B1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0123918, upper bound: 0.0131004
NS_A1_A1_B2_A2_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124504, upper bound: 0.0130975
NS_A1_A1_B2_A2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124505, upper bound: 0.0130718
NS_A1_A1_B2_A2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124504, upper bound: 0.0131532
NS_A1_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125726, upper bound: 0.0133895
NS_A1_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125725, upper bound: 0.0134476
NS_A1_A2_B2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124587, upper bound: 0.0134591
NS_A1_A2_B2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124587, upper bound: 0.0135017
NS_A1_A2_B2_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0134592
NS_A1_A2_B2_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0135017
NS_A1_A2_B2_A2_A2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0123405, upper bound: 0.0131746
NS_A1_A2_B2_A2_A2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0123866, upper bound: 0.0131586
NS_A1_A2_B2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0134732
NS_A1_A2_B2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124569, upper bound: 0.0135271
NS_A2_A1_B2_A2_B1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124749, upper bound: 0.0131004
NS_A2_A1_B2_A2_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125101, upper bound: 0.0130975
NS_A2_A1_B2_A2_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0124749, upper bound: 0.0130978
NS_A2_A1_B2_A2_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125101, upper bound: 0.0130953
NS_A2_A2_B2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0134591
NS_A2_A2_B2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0135017
NS_A2_A2_B2_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125600, upper bound: 0.0134591
NS_A2_A2_B2_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125600, upper bound: 0.0135017
NS_A2_A2_B2_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0134507
NS_A2_A2_B2_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125810, upper bound: 0.0134992
NS_A2_A2_B2_A2_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125601, upper bound: 0.0134507
NS_A2_A2_B2_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 5.64
Output dim: 0, lower bound: -0.0125601, upper bound: 0.0134992

## BFS NS instance: NS_A1_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9900260, 1.0078346, 0.9890388, 1.0084298, -0.0184038, 0.0187957
1: -0.0037492, 0.0006658, -0.0039952, 0.0007268, -0.0044760, 0.0046609
2: -0.0136298, 0.0098148, -0.0141385, 0.0111184, -0.0247482, 0.0239533
3: -0.0057404, 0.0049089, -0.0063337, 0.0050562, -0.0107966, 0.0112426
4: -0.0021009, 0.0024275, -0.0021635, 0.0026798, -0.0047807, 0.0045911
5: -0.0181233, 0.0113039, -0.0185304, 0.0129435, -0.0285199, 0.0275356
6: -0.0013282, 0.0062343, -0.0017444, 0.0067013, -0.0080295, 0.0079786
7: -0.0065741, 0.0127503, -0.0076508, 0.0130176, -0.0195917, 0.0204011
8: -0.0030214, 0.0071411, -0.0035876, 0.0072817, -0.0103031, 0.0107288
9: -0.0101443, 0.0016396, -0.0103073, 0.0022962, -0.0124405, 0.0119470

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121862, upper bound: 0.0130328
time: 2.43 seconds

## Relational analysis of NS_A1_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121694, upper bound: 0.0131017
time: 2.38 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9904216, 1.0069635, -0.0162934, 0.0173978
1: -0.0039877, 0.0006642, -0.0036507, 0.0004712, -0.0044588, 0.0040849
2: -0.0136169, 0.0110785, -0.0125510, 0.0092925, -0.0229094, 0.0215153
3: -0.0063156, 0.0049052, -0.0055027, 0.0044395, -0.0107551, 0.0098531
4: -0.0020993, 0.0026721, -0.0019013, 0.0023264, -0.0041898, 0.0045734
5: -0.0181130, 0.0128933, -0.0168264, 0.0106470, -0.0272270, 0.0273068
6: -0.0017316, 0.0062224, -0.0011615, 0.0058115, -0.0068683, 0.0073839
7: -0.0076179, 0.0127435, -0.0061428, 0.0118986, -0.0195165, 0.0178796
8: -0.0035703, 0.0071376, -0.0027946, 0.0066932, -0.0102635, 0.0094027
9: -0.0101402, 0.0022761, -0.0096250, 0.0013766, -0.0109029, 0.0119011

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120453, upper bound: 0.0130881
time: 2.45 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120659, upper bound: 0.0130687
time: 2.39 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9900260, 1.0078346, -0.0187655, 0.0177934
1: -0.0039877, 0.0006642, -0.0037492, 0.0006658, -0.0046534, 0.0044134
2: -0.0136169, 0.0110785, -0.0136298, 0.0098148, -0.0234317, 0.0247083
3: -0.0063156, 0.0049052, -0.0057404, 0.0049089, -0.0112245, 0.0106456
4: -0.0020993, 0.0026721, -0.0021009, 0.0024275, -0.0045269, 0.0047730
5: -0.0181130, 0.0128933, -0.0181233, 0.0113039, -0.0271126, 0.0280568
6: -0.0017316, 0.0062224, -0.0013282, 0.0062343, -0.0079659, 0.0075506
7: -0.0076179, 0.0127435, -0.0065741, 0.0127503, -0.0203682, 0.0193177
8: -0.0035703, 0.0071376, -0.0030214, 0.0071411, -0.0107114, 0.0101590
9: -0.0101402, 0.0022761, -0.0101443, 0.0016396, -0.0117798, 0.0124204

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120454, upper bound: 0.0131442
time: 2.42 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B1_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120659, upper bound: 0.0131232
time: 2.46 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9894437, 1.0068061, -0.0158717, 0.0183757
1: -0.0039877, 0.0006642, -0.0038943, 0.0004320, -0.0044196, 0.0042664
2: -0.0136169, 0.0110785, -0.0123431, 0.0105838, -0.0242006, 0.0209584
3: -0.0063156, 0.0049052, -0.0060904, 0.0043449, -0.0106605, 0.0102910
4: -0.0020993, 0.0026721, -0.0018611, 0.0025764, -0.0043761, 0.0045332
5: -0.0181130, 0.0128933, -0.0165650, 0.0122711, -0.0284371, 0.0266412
6: -0.0017316, 0.0062224, -0.0015737, 0.0057452, -0.0066905, 0.0077961
7: -0.0076179, 0.0127435, -0.0072092, 0.0117269, -0.0193448, 0.0186742
8: -0.0035703, 0.0071376, -0.0033554, 0.0066029, -0.0101733, 0.0098206
9: -0.0101402, 0.0022761, -0.0095203, 0.0020269, -0.0113875, 0.0117964

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120724, upper bound: 0.0129652
time: 2.45 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120640, upper bound: 0.0130687
time: 2.41 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9891176, 1.0074958, -0.0160817, 0.0187018
1: -0.0039877, 0.0006642, -0.0039756, 0.0006038, -0.0045915, 0.0042296
2: -0.0136169, 0.0110785, -0.0132539, 0.0110144, -0.0246313, 0.0212357
3: -0.0063156, 0.0049052, -0.0062864, 0.0047595, -0.0110751, 0.0102021
4: -0.0020993, 0.0026721, -0.0020374, 0.0026597, -0.0043383, 0.0047095
5: -0.0181130, 0.0128933, -0.0177104, 0.0128127, -0.0281916, 0.0272650
6: -0.0017316, 0.0062224, -0.0017112, 0.0060359, -0.0067790, 0.0079336
7: -0.0076179, 0.0127435, -0.0075649, 0.0124792, -0.0200970, 0.0185130
8: -0.0035703, 0.0071376, -0.0035425, 0.0069985, -0.0105688, 0.0097358
9: -0.0101402, 0.0022761, -0.0099790, 0.0022438, -0.0112892, 0.0122551

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120724, upper bound: 0.0130284
time: 2.34 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0120640, upper bound: 0.0131232
time: 2.44 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9893871, 1.0070686, -0.0165055, 0.0184323
1: -0.0039877, 0.0006642, -0.0039084, 0.0004974, -0.0044850, 0.0043433
2: -0.0136169, 0.0110785, -0.0126898, 0.0106584, -0.0242753, 0.0217953
3: -0.0063156, 0.0049052, -0.0061244, 0.0045027, -0.0108183, 0.0104765
4: -0.0020993, 0.0026721, -0.0019282, 0.0025908, -0.0044550, 0.0046003
5: -0.0181130, 0.0128933, -0.0170010, 0.0123649, -0.0289499, 0.0273503
6: -0.0017316, 0.0062224, -0.0015975, 0.0058559, -0.0069576, 0.0078200
7: -0.0076179, 0.0127435, -0.0072709, 0.0120133, -0.0196312, 0.0190110
8: -0.0035703, 0.0071376, -0.0033878, 0.0067535, -0.0103238, 0.0099977
9: -0.0101402, 0.0022761, -0.0096949, 0.0020645, -0.0115928, 0.0119710

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122580, upper bound: 0.0131700
time: 2.40 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123382, upper bound: 0.0133536
time: 2.36 seconds

## BFS NS instance: NS_A1_A2_B2_A2_A2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890690, 1.0078194, 0.9890963, 1.0078416, -0.0187725, 0.0187231
1: -0.0039877, 0.0006642, -0.0039808, 0.0006665, -0.0046541, 0.0046451
2: -0.0136169, 0.0110785, -0.0136358, 0.0110424, -0.0246593, 0.0247143
3: -0.0063156, 0.0049052, -0.0062992, 0.0049106, -0.0112262, 0.0112043
4: -0.0020993, 0.0026721, -0.0021017, 0.0026651, -0.0047645, 0.0047738
5: -0.0181130, 0.0128933, -0.0181282, 0.0128479, -0.0286629, 0.0279004
6: -0.0017316, 0.0062224, -0.0017201, 0.0062398, -0.0079714, 0.0079425
7: -0.0076179, 0.0127435, -0.0075881, 0.0127535, -0.0203714, 0.0203316
8: -0.0035703, 0.0071376, -0.0035546, 0.0071428, -0.0107131, 0.0106922
9: -0.0101402, 0.0022761, -0.0101463, 0.0022579, -0.0123981, 0.0124224

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122580, upper bound: 0.0131962
time: 2.33 seconds

## Relational analysis of NS_A1_A2_B2_A2_A2_B2_B2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_A2_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0123383, upper bound: 0.0134169
time: 2.40 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9904216, 1.0069635, -0.0168251, 0.0180588
1: -0.0039966, 0.0007320, -0.0036507, 0.0004712, -0.0044678, 0.0041553
2: -0.0141817, 0.0111258, -0.0125510, 0.0092925, -0.0234743, 0.0222173
3: -0.0063371, 0.0050687, -0.0055027, 0.0044395, -0.0107767, 0.0100229
4: -0.0021689, 0.0026813, -0.0019013, 0.0023264, -0.0042621, 0.0045826
5: -0.0185650, 0.0129528, -0.0168264, 0.0106470, -0.0276964, 0.0277193
6: -0.0017467, 0.0067410, -0.0011615, 0.0058115, -0.0070924, 0.0079025
7: -0.0076569, 0.0130403, -0.0061428, 0.0118986, -0.0195555, 0.0181878
8: -0.0035908, 0.0072936, -0.0027946, 0.0066932, -0.0102841, 0.0095648
9: -0.0103212, 0.0022999, -0.0096250, 0.0013766, -0.0110908, 0.0119249

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121766, upper bound: 0.0130881
time: 2.28 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121866, upper bound: 0.0130687
time: 2.56 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9900260, 1.0078346, -0.0188013, 0.0184544
1: -0.0039966, 0.0007320, -0.0037492, 0.0006658, -0.0046623, 0.0044812
2: -0.0141817, 0.0111258, -0.0136298, 0.0098148, -0.0239966, 0.0247556
3: -0.0063371, 0.0050687, -0.0057404, 0.0049089, -0.0112460, 0.0108091
4: -0.0021689, 0.0026813, -0.0021009, 0.0024275, -0.0045964, 0.0047822
5: -0.0185650, 0.0129528, -0.0181233, 0.0113039, -0.0275700, 0.0285299
6: -0.0017467, 0.0067410, -0.0013282, 0.0062343, -0.0079810, 0.0080692
7: -0.0076569, 0.0130403, -0.0065741, 0.0127503, -0.0204072, 0.0196145
8: -0.0035908, 0.0072936, -0.0030214, 0.0071411, -0.0107320, 0.0103151
9: -0.0103212, 0.0022999, -0.0101443, 0.0016396, -0.0119608, 0.0124442

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121766, upper bound: 0.0131441
time: 2.46 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B1_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121866, upper bound: 0.0131232
time: 2.49 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9894437, 1.0068061, -0.0165383, 0.0190367
1: -0.0039966, 0.0007320, -0.0038943, 0.0004320, -0.0044285, 0.0043933
2: -0.0141817, 0.0111258, -0.0123431, 0.0105838, -0.0247655, 0.0218386
3: -0.0063371, 0.0050687, -0.0060904, 0.0043449, -0.0106821, 0.0105970
4: -0.0021689, 0.0026813, -0.0018611, 0.0025764, -0.0045062, 0.0045424
5: -0.0185650, 0.0129528, -0.0165650, 0.0122711, -0.0292829, 0.0271644
6: -0.0017467, 0.0067410, -0.0015737, 0.0057452, -0.0069715, 0.0083147
7: -0.0076569, 0.0130403, -0.0072092, 0.0117269, -0.0193839, 0.0192296
8: -0.0035908, 0.0072936, -0.0033554, 0.0066029, -0.0101938, 0.0101127
9: -0.0103212, 0.0022999, -0.0095203, 0.0020269, -0.0117261, 0.0118202

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122567, upper bound: 0.0132171
time: 2.38 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124558, upper bound: 0.0133297
time: 2.51 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9891176, 1.0074958, -0.0168328, 0.0193628
1: -0.0039966, 0.0007320, -0.0039756, 0.0006038, -0.0046004, 0.0043754
2: -0.0141817, 0.0111258, -0.0132539, 0.0110144, -0.0251962, 0.0222275
3: -0.0063371, 0.0050687, -0.0062864, 0.0047595, -0.0110966, 0.0105538
4: -0.0021689, 0.0026813, -0.0020374, 0.0026597, -0.0044879, 0.0047186
5: -0.0185650, 0.0129528, -0.0177104, 0.0128127, -0.0291635, 0.0278879
6: -0.0017467, 0.0067410, -0.0017112, 0.0060359, -0.0070956, 0.0084522
7: -0.0076569, 0.0130403, -0.0075649, 0.0124792, -0.0201361, 0.0191513
8: -0.0035908, 0.0072936, -0.0035425, 0.0069985, -0.0105894, 0.0100715
9: -0.0103212, 0.0022999, -0.0099790, 0.0022438, -0.0116783, 0.0122789

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122567, upper bound: 0.0132462
time: 2.54 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124558, upper bound: 0.0133820
time: 3.09 seconds

## BFS NS instance: NS_A2_A2_B2_A2_A2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.9890332, 1.0084804, 0.9902167, 1.0073375, -0.0172507, 0.0182638
1: -0.0039966, 0.0007320, -0.0037017, 0.0005643, -0.0045609, 0.0041747
2: -0.0141817, 0.0111258, -0.0130447, 0.0095631, -0.0237448, 0.0227794
3: -0.0063371, 0.0050687, -0.0056258, 0.0046643, -0.0110014, 0.0100697
4: -0.0021689, 0.0026813, -0.0019969, 0.0023788, -0.0042820, 0.0046782
5: -0.0185650, 0.0129528, -0.0174474, 0.0109873, -0.0278258, 0.0283083
6: -0.0017467, 0.0067410, -0.0012479, 0.0059692, -0.0072718, 0.0079889
7: -0.0076569, 0.0130403, -0.0063662, 0.0123064, -0.0199633, 0.0182728
8: -0.0035908, 0.0072936, -0.0029121, 0.0069077, -0.0104985, 0.0096095
9: -0.0103212, 0.0022999, -0.0098736, 0.0015128, -0.0111427, 0.0121736

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121766, upper bound: 0.0130834
time: 2.18 seconds

## Relational analysis of NS_A2_A2_B2_A2_A2_B2_B1_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_A2_B2_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121866, upper bound: 0.0130637
time: 2.43 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.02 + 598.39 = 602.40 seconds
