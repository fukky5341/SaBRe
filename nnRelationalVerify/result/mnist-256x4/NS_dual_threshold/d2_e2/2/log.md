## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.07289625500000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835)
1: (0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128)
2: (-0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145)
3: (-0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669)
4: (-0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407)
5: (-0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920)
6: (-0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919)
7: (-0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113)
8: (-0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782)
9: (-0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 2.82 = 4.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0767329, upper bound: 0.0767329

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0766297, upper bound: 0.0766600
time: 2.01 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0766149, upper bound: 0.0766149
time: 1.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.90 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.90
Output dim: 1, lower bound: -0.0766297, upper bound: 0.0766600
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.90
Output dim: 1, lower bound: -0.0766149, upper bound: 0.0766149

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0551280, 0.0106954, -0.0557566, 0.0117566, -0.0668845, 0.0664520
1: 0.9248901, 1.0358126, 0.9180285, 1.0372927, -0.1124026, 0.1177841
2: -0.0171014, 0.0546069, -0.0171906, 0.0569246, -0.0740260, 0.0717975
3: -0.0388624, 0.0084467, -0.0394698, 0.0086630, -0.0475254, 0.0479165
4: -0.0419033, 0.0236819, -0.0428787, 0.0240081, -0.0659113, 0.0665606
5: -0.0071963, 0.0695283, -0.0076924, 0.0706329, -0.0778292, 0.0772208
6: -0.0108317, 0.0219962, -0.0110127, 0.0225480, -0.0333797, 0.0330089
7: -0.0115818, 0.0982913, -0.0116636, 0.1008392, -0.1124211, 0.1099550
8: -0.0124437, 0.0332155, -0.0132191, 0.0338024, -0.0462461, 0.0464346
9: -0.0610026, 0.0245486, -0.0619285, 0.0256569, -0.0866596, 0.0864772

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0766149, upper bound: 0.0766150
time: 2.32 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0766149, upper bound: 0.0766149
time: 1.67 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0559409, 0.0128184, -0.0557518, 0.0117440, -0.0676849, 0.0685703
1: 0.9111223, 1.0377848, 0.9181098, 1.0372815, -0.1261593, 0.1196750
2: -0.0173084, 0.0587941, -0.0171894, 0.0569011, -0.0742095, 0.0759835
3: -0.0396623, 0.0089041, -0.0394651, 0.0086603, -0.0483226, 0.0483692
4: -0.0430642, 0.0244121, -0.0428716, 0.0240038, -0.0670680, 0.0672837
5: -0.0075138, 0.0716626, -0.0076896, 0.0706221, -0.0781359, 0.0793522
6: -0.0112311, 0.0223132, -0.0110103, 0.0225453, -0.0337764, 0.0333235
7: -0.0117435, 0.1028558, -0.0116627, 0.1008135, -0.1125570, 0.1145185
8: -0.0135341, 0.0343964, -0.0132124, 0.0337954, -0.0473296, 0.0476088
9: -0.0626421, 0.0268140, -0.0619207, 0.0256435, -0.0882856, 0.0887347

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0760655, upper bound: 0.0753563
time: 1.45 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762647, upper bound: 0.0762647
time: 2.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.46 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.46
Output dim: 1, lower bound: -0.0766149, upper bound: 0.0766150
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.46
Output dim: 1, lower bound: -0.0766149, upper bound: 0.0766149
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.46
Output dim: 1, lower bound: -0.0760655, upper bound: 0.0753563
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.46
Output dim: 1, lower bound: -0.0762647, upper bound: 0.0762647

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0551280, 0.0106954, -0.0551280, 0.0106954, -0.0658233, 0.0658233
1: 0.9248901, 1.0358126, 0.9248901, 1.0358126, -0.1109225, 0.1109225
2: -0.0171014, 0.0546069, -0.0171014, 0.0546069, -0.0717082, 0.0717082
3: -0.0388624, 0.0084467, -0.0388624, 0.0084467, -0.0473091, 0.0473091
4: -0.0419033, 0.0236819, -0.0419033, 0.0236819, -0.0655851, 0.0655851
5: -0.0071963, 0.0695283, -0.0071963, 0.0695283, -0.0767246, 0.0767246
6: -0.0108317, 0.0219962, -0.0108317, 0.0219962, -0.0328279, 0.0328279
7: -0.0115818, 0.0982913, -0.0115818, 0.0982913, -0.1098731, 0.1098731
8: -0.0124437, 0.0332155, -0.0124437, 0.0332155, -0.0456592, 0.0456592
9: -0.0610026, 0.0245486, -0.0610026, 0.0245486, -0.0855513, 0.0855513

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0753671, upper bound: 0.0760945
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762814, upper bound: 0.0763017
time: 2.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0551280, 0.0106954, -0.0559409, 0.0128184, -0.0679464, 0.0666362
1: 0.9248901, 1.0358126, 0.9111223, 1.0377848, -0.1128947, 0.1246904
2: -0.0171014, 0.0546069, -0.0173084, 0.0587941, -0.0758954, 0.0719153
3: -0.0388624, 0.0084467, -0.0396623, 0.0089041, -0.0477665, 0.0481090
4: -0.0419033, 0.0236819, -0.0430642, 0.0244121, -0.0663154, 0.0667461
5: -0.0071963, 0.0695283, -0.0075138, 0.0716626, -0.0788589, 0.0770422
6: -0.0108317, 0.0219962, -0.0112311, 0.0223132, -0.0331449, 0.0332274
7: -0.0115818, 0.0982913, -0.0117435, 0.1028558, -0.1144376, 0.1100349
8: -0.0124437, 0.0332155, -0.0135341, 0.0343964, -0.0468401, 0.0467497
9: -0.0610026, 0.0245486, -0.0626421, 0.0268140, -0.0878167, 0.0871907

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0753671, upper bound: 0.0760945
time: 1.44 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762814, upper bound: 0.0763017
time: 2.08 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0487823, 0.0096347, -0.0549010, 0.0107721, -0.0595545, 0.0645357
1: 0.9315191, 1.0207678, 0.9246511, 1.0352560, -0.1037369, 0.0961167
2: -0.0169818, 0.0434946, -0.0171477, 0.0542586, -0.0712405, 0.0606422
3: -0.0331020, 0.0082104, -0.0386839, 0.0084898, -0.0415917, 0.0468943
4: -0.0341849, 0.0232785, -0.0418094, 0.0238079, -0.0579927, 0.0650879
5: -0.0068704, 0.0599902, -0.0076041, 0.0689895, -0.0758599, 0.0675942
6: -0.0105842, 0.0218089, -0.0109280, 0.0224741, -0.0330583, 0.0327370
7: -0.0114652, 0.0841483, -0.0116277, 0.0976987, -0.1091639, 0.0957760
8: -0.0077873, 0.0326447, -0.0124923, 0.0332409, -0.0410282, 0.0451370
9: -0.0485412, 0.0234303, -0.0602560, 0.0246490, -0.0731902, 0.0836864

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0726637, upper bound: 0.0738225
time: 1.38 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755692, upper bound: 0.0748548
time: 1.50 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0530477, 0.0097634, -0.0557518, 0.0117440, -0.0647917, 0.0655153
1: 0.9313115, 1.0308919, 0.9181098, 1.0372815, -0.1059700, 0.1127821
2: -0.0170963, 0.0502325, -0.0171894, 0.0569011, -0.0739975, 0.0674218
3: -0.0370045, 0.0083090, -0.0394651, 0.0086603, -0.0456649, 0.0477741
4: -0.0394429, 0.0235854, -0.0428716, 0.0240038, -0.0634466, 0.0664571
5: -0.0071892, 0.0662518, -0.0076896, 0.0706221, -0.0778113, 0.0739414
6: -0.0108183, 0.0220466, -0.0110103, 0.0225453, -0.0333636, 0.0330569
7: -0.0115728, 0.0927418, -0.0116627, 0.1008135, -0.1123863, 0.1044045
8: -0.0109685, 0.0326753, -0.0132124, 0.0337954, -0.0447639, 0.0458877
9: -0.0569951, 0.0236242, -0.0619207, 0.0256435, -0.0826386, 0.0855449

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728353, upper bound: 0.0747848
time: 1.84 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0757563, upper bound: 0.0757563
time: 1.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.06 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0753671, upper bound: 0.0760945
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0762814, upper bound: 0.0763017
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0753671, upper bound: 0.0760945
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0762814, upper bound: 0.0763017
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0726637, upper bound: 0.0738225
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0755692, upper bound: 0.0748548
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0728353, upper bound: 0.0747848
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.06
Output dim: 1, lower bound: -0.0757563, upper bound: 0.0757563

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0542769, 0.0097343, -0.0479483, 0.0094258, -0.0637027, 0.0576826
1: 0.9313573, 1.0337857, 0.9319175, 1.0187615, -0.0874042, 0.1018682
2: -0.0170597, 0.0519698, -0.0167633, 0.0419881, -0.0590478, 0.0687331
3: -0.0380805, 0.0082777, -0.0322861, 0.0080218, -0.0461023, 0.0405638
4: -0.0408396, 0.0234870, -0.0329895, 0.0226909, -0.0635305, 0.0564764
5: -0.0071071, 0.0678983, -0.0065266, 0.0587051, -0.0658123, 0.0744248
6: -0.0107496, 0.0219217, -0.0101628, 0.0214652, -0.0322147, 0.0320845
7: -0.0115470, 0.0951831, -0.0112948, 0.0824653, -0.0940123, 0.1064779
8: -0.0117281, 0.0326672, -0.0066862, 0.0325860, -0.0443142, 0.0393534
9: -0.0593385, 0.0235647, -0.0469109, 0.0230581, -0.0823966, 0.0704756

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738797, upper bound: 0.0726879
time: 1.41 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749171, upper bound: 0.0756362
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0551280, 0.0106954, -0.0522289, 0.0095718, -0.0646998, 0.0629243
1: 0.9248901, 1.0358126, 0.9316761, 1.0289143, -0.1040242, 0.1041365
2: -0.0171014, 0.0546069, -0.0168964, 0.0487296, -0.0658309, 0.0715032
3: -0.0388624, 0.0084467, -0.0362013, 0.0081364, -0.0469988, 0.0446480
4: -0.0419033, 0.0236819, -0.0382744, 0.0230478, -0.0649511, 0.0619562
5: -0.0071963, 0.0695283, -0.0068663, 0.0649361, -0.0721324, 0.0763946
6: -0.0108317, 0.0219962, -0.0104324, 0.0217249, -0.0325566, 0.0324286
7: -0.0115818, 0.0982913, -0.0114165, 0.0910789, -0.1026607, 0.1097078
8: -0.0124437, 0.0332155, -0.0098924, 0.0326216, -0.0450653, 0.0431079
9: -0.0610026, 0.0245486, -0.0553571, 0.0232837, -0.0842864, 0.0799057

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748385, upper bound: 0.0728547
time: 1.61 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0758215, upper bound: 0.0758215
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0542769, 0.0097343, -0.0487823, 0.0096347, -0.0639116, 0.0585166
1: 0.9313573, 1.0337857, 0.9315191, 1.0207678, -0.0894105, 0.1022666
2: -0.0170597, 0.0519698, -0.0169818, 0.0434946, -0.0605542, 0.0689517
3: -0.0380805, 0.0082777, -0.0331020, 0.0082104, -0.0462909, 0.0413797
4: -0.0408396, 0.0234870, -0.0341849, 0.0232785, -0.0641181, 0.0576719
5: -0.0071071, 0.0678983, -0.0068704, 0.0599902, -0.0670973, 0.0747686
6: -0.0107496, 0.0219217, -0.0105842, 0.0218089, -0.0325585, 0.0325059
7: -0.0115470, 0.0951831, -0.0114652, 0.0841483, -0.0956953, 0.1066483
8: -0.0117281, 0.0326672, -0.0077873, 0.0326447, -0.0443728, 0.0404545
9: -0.0593385, 0.0235647, -0.0485412, 0.0234303, -0.0827689, 0.0721058

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738293, upper bound: 0.0726661
time: 1.63 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748679, upper bound: 0.0755978
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0551280, 0.0106954, -0.0530477, 0.0097634, -0.0648914, 0.0637431
1: 0.9248901, 1.0358126, 0.9313115, 1.0308919, -0.1060018, 0.1045011
2: -0.0171014, 0.0546069, -0.0170963, 0.0502325, -0.0673338, 0.0717032
3: -0.0388624, 0.0084467, -0.0370045, 0.0083090, -0.0471714, 0.0454513
4: -0.0419033, 0.0236819, -0.0394429, 0.0235854, -0.0654887, 0.0631247
5: -0.0071963, 0.0695283, -0.0071892, 0.0662518, -0.0734481, 0.0767175
6: -0.0108317, 0.0219962, -0.0108183, 0.0220466, -0.0328783, 0.0328145
7: -0.0115818, 0.0982913, -0.0115728, 0.0927418, -0.1043236, 0.1098642
8: -0.0124437, 0.0332155, -0.0109685, 0.0326753, -0.0451190, 0.0441840
9: -0.0610026, 0.0245486, -0.0569951, 0.0236242, -0.0846269, 0.0815437

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747983, upper bound: 0.0728399
time: 2.40 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0757714, upper bound: 0.0757913
time: 1.49 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0462473, 0.0095912, -0.0392446, 0.0095769, -0.0558242, 0.0488357
1: 0.9315658, 1.0148174, 0.9314365, 1.0033147, -0.0717490, 0.0833808
2: -0.0169556, 0.0396722, -0.0170239, 0.0293837, -0.0463393, 0.0566961
3: -0.0308222, 0.0081880, -0.0245802, 0.0082479, -0.0390701, 0.0327682
4: -0.0310927, 0.0232089, -0.0227823, 0.0233963, -0.0544890, 0.0459913
5: -0.0066177, 0.0565141, -0.0061754, 0.0470222, -0.0536399, 0.0626895
6: -0.0105209, 0.0215933, -0.0105976, 0.0212734, -0.0317943, 0.0321908
7: -0.0114271, 0.0791608, -0.0114045, 0.0653288, -0.0767559, 0.0905653
8: -0.0070860, 0.0326378, -0.0072409, 0.0326569, -0.0397428, 0.0398788
9: -0.0437372, 0.0233867, -0.0304053, 0.0235075, -0.0672447, 0.0537920

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0716262, upper bound: 0.0714201
time: 1.60 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721845, upper bound: 0.0733051
time: 1.55 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0487823, 0.0096347, -0.0513380, 0.0097729, -0.0585552, 0.0609727
1: 0.9315191, 1.0207678, 0.9312866, 1.0268230, -0.0953040, 0.0894812
2: -0.0169818, 0.0434946, -0.0171099, 0.0475433, -0.0645251, 0.0606045
3: -0.0331020, 0.0082104, -0.0354672, 0.0083208, -0.0414227, 0.0436776
4: -0.0341849, 0.0232785, -0.0374270, 0.0236222, -0.0578070, 0.0607055
5: -0.0068704, 0.0599902, -0.0072510, 0.0636604, -0.0705307, 0.0672412
6: -0.0105842, 0.0218089, -0.0108421, 0.0221737, -0.0327579, 0.0326510
7: -0.0114652, 0.0841483, -0.0115800, 0.0892704, -0.1007356, 0.0957283
8: -0.0077873, 0.0326447, -0.0098335, 0.0326790, -0.0404662, 0.0424782
9: -0.0485412, 0.0234303, -0.0535003, 0.0236476, -0.0721887, 0.0769306

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732638, upper bound: 0.0739852
time: 1.97 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0750134, upper bound: 0.0742699
time: 1.80 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0505642, 0.0097246, -0.0400910, 0.0096262, -0.0601904, 0.0498156
1: 0.9313586, 1.0250183, 0.9313619, 1.0034695, -0.0721109, 0.0936564
2: -0.0170701, 0.0464689, -0.0170652, 0.0306998, -0.0477699, 0.0635340
3: -0.0347641, 0.0082865, -0.0253549, 0.0082834, -0.0430475, 0.0336415
4: -0.0363942, 0.0235155, -0.0238216, 0.0235067, -0.0599009, 0.0473372
5: -0.0069396, 0.0628194, -0.0062580, 0.0482436, -0.0551832, 0.0690774
6: -0.0107582, 0.0218340, -0.0106839, 0.0213434, -0.0321016, 0.0325180
7: -0.0115393, 0.0878555, -0.0114461, 0.0670362, -0.0785755, 0.0993016
8: -0.0091600, 0.0326684, -0.0073808, 0.0326679, -0.0418278, 0.0400492
9: -0.0522943, 0.0235803, -0.0320627, 0.0235772, -0.0758715, 0.0556431

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0718406, upper bound: 0.0721557
time: 1.75 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0723536, upper bound: 0.0742843
time: 1.45 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0530477, 0.0097634, -0.0521772, 0.0098160, -0.0628637, 0.0619406
1: 0.9313115, 1.0308919, 0.9312109, 1.0288180, -0.0975065, 0.0996810
2: -0.0170963, 0.0502325, -0.0171515, 0.0488709, -0.0659672, 0.0673840
3: -0.0370045, 0.0083090, -0.0362372, 0.0083566, -0.0453611, 0.0445462
4: -0.0394429, 0.0235854, -0.0384722, 0.0237337, -0.0631766, 0.0620576
5: -0.0071892, 0.0662518, -0.0073358, 0.0648806, -0.0720698, 0.0735876
6: -0.0108183, 0.0220466, -0.0109245, 0.0222444, -0.0330627, 0.0329711
7: -0.0115728, 0.0927418, -0.0116156, 0.0909573, -0.1025302, 0.1043574
8: -0.0109685, 0.0326753, -0.0105137, 0.0326901, -0.0436586, 0.0431890
9: -0.0569951, 0.0236242, -0.0551477, 0.0237182, -0.0807133, 0.0787719

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748647, upper bound: 0.0734089
time: 1.95 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752067, upper bound: 0.0752067
time: 1.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.97 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0738797, upper bound: 0.0726879
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0749171, upper bound: 0.0756362
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0748385, upper bound: 0.0728547
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0758215, upper bound: 0.0758215
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0738293, upper bound: 0.0726661
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0748679, upper bound: 0.0755978
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0747983, upper bound: 0.0728399
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0757714, upper bound: 0.0757913
NS_A2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0716262, upper bound: 0.0714201
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0721845, upper bound: 0.0733051
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0732638, upper bound: 0.0739852
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0750134, upper bound: 0.0742699
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0718406, upper bound: 0.0721557
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0723536, upper bound: 0.0742843
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0748647, upper bound: 0.0734089
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.97
Output dim: 1, lower bound: -0.0752067, upper bound: 0.0752067

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0386245, 0.0094811, -0.0453940, 0.0093814, -0.0480059, 0.0548751
1: 0.9316024, 1.0025718, 0.9319657, 1.0127878, -0.0811854, 0.0706061
2: -0.0169327, 0.0284094, -0.0167363, 0.0381430, -0.0550757, 0.0451457
3: -0.0239849, 0.0081692, -0.0299877, 0.0079987, -0.0319836, 0.0381569
4: -0.0218442, 0.0231513, -0.0298714, 0.0226192, -0.0444634, 0.0530227
5: -0.0056231, 0.0463104, -0.0062736, 0.0552099, -0.0608330, 0.0525841
6: -0.0104157, 0.0206736, -0.0100979, 0.0212484, -0.0316641, 0.0307715
7: -0.0113252, 0.0642233, -0.0112561, 0.0774429, -0.0887681, 0.0754793
8: -0.0069419, 0.0326324, -0.0063838, 0.0325789, -0.0395209, 0.0390162
9: -0.0294906, 0.0233525, -0.0420846, 0.0230131, -0.0525038, 0.0654371

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715093, upper bound: 0.0716613
time: 2.01 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733645, upper bound: 0.0722038
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0507102, 0.0096763, -0.0479483, 0.0094258, -0.0601359, 0.0576245
1: 0.9314473, 1.0253417, 0.9319175, 1.0187615, -0.0873142, 0.0934242
2: -0.0170213, 0.0465411, -0.0167633, 0.0419881, -0.0590094, 0.0633044
3: -0.0348589, 0.0082444, -0.0322861, 0.0080218, -0.0428807, 0.0405305
4: -0.0364481, 0.0233845, -0.0329895, 0.0226909, -0.0591390, 0.0563740
5: -0.0067404, 0.0629642, -0.0065266, 0.0587051, -0.0654455, 0.0694907
6: -0.0106630, 0.0216101, -0.0101628, 0.0214652, -0.0321281, 0.0317729
7: -0.0114996, 0.0881484, -0.0112948, 0.0824653, -0.0939649, 0.0994432
8: -0.0090915, 0.0326553, -0.0066862, 0.0325860, -0.0416776, 0.0393415
9: -0.0525798, 0.0234974, -0.0469109, 0.0230581, -0.0756379, 0.0704083

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0740396, upper bound: 0.0733273
time: 1.52 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0743315, upper bound: 0.0750819
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0394707, 0.0095301, -0.0496808, 0.0095320, -0.0490027, 0.0592109
1: 0.9315279, 1.0027279, 0.9317241, 1.0228852, -0.0913574, 0.0710037
2: -0.0169739, 0.0297220, -0.0168695, 0.0448721, -0.0618460, 0.0465915
3: -0.0247587, 0.0082047, -0.0338971, 0.0081134, -0.0328721, 0.0421018
4: -0.0228805, 0.0232616, -0.0351390, 0.0229763, -0.0458568, 0.0584006
5: -0.0057086, 0.0475331, -0.0066095, 0.0614230, -0.0671315, 0.0541426
6: -0.0105018, 0.0207460, -0.0103709, 0.0215063, -0.0320080, 0.0311169
7: -0.0113665, 0.0659306, -0.0113821, 0.0860578, -0.0974243, 0.0773127
8: -0.0070815, 0.0326434, -0.0080584, 0.0326145, -0.0396960, 0.0407018
9: -0.0311504, 0.0234222, -0.0505405, 0.0232388, -0.0543892, 0.0739626

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722271, upper bound: 0.0718723
time: 1.82 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0743420, upper bound: 0.0723735
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0515498, 0.0097192, -0.0522289, 0.0095718, -0.0611217, 0.0619481
1: 0.9313717, 1.0273391, 0.9316761, 1.0289143, -0.0975426, 0.0956630
2: -0.0170629, 0.0478696, -0.0168964, 0.0487296, -0.0657924, 0.0647659
3: -0.0356297, 0.0082802, -0.0362013, 0.0081364, -0.0437661, 0.0444815
4: -0.0374945, 0.0234960, -0.0382744, 0.0230478, -0.0605423, 0.0617704
5: -0.0068287, 0.0641849, -0.0068663, 0.0649361, -0.0717648, 0.0710512
6: -0.0107453, 0.0216836, -0.0104324, 0.0217249, -0.0324702, 0.0321160
7: -0.0115350, 0.0898363, -0.0114165, 0.0910789, -0.1026139, 0.1012528
8: -0.0097660, 0.0326664, -0.0098924, 0.0326216, -0.0423876, 0.0425588
9: -0.0542278, 0.0235679, -0.0553571, 0.0232837, -0.0775115, 0.0789250

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734754, upper bound: 0.0749455
time: 1.47 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752804, upper bound: 0.0752804
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0386245, 0.0094811, -0.0462473, 0.0095912, -0.0482156, 0.0557284
1: 0.9316024, 1.0025718, 0.9315658, 1.0148174, -0.0832149, 0.0710061
2: -0.0169327, 0.0284094, -0.0169556, 0.0396722, -0.0566048, 0.0453650
3: -0.0239849, 0.0081692, -0.0308222, 0.0081880, -0.0321729, 0.0389914
4: -0.0218442, 0.0231513, -0.0310927, 0.0232089, -0.0450532, 0.0542440
5: -0.0056231, 0.0463104, -0.0066177, 0.0565141, -0.0621372, 0.0529281
6: -0.0104157, 0.0206736, -0.0105209, 0.0215933, -0.0320089, 0.0311945
7: -0.0113252, 0.0642233, -0.0114271, 0.0791608, -0.0904860, 0.0756504
8: -0.0069419, 0.0326324, -0.0070860, 0.0326378, -0.0395798, 0.0397184
9: -0.0294906, 0.0233525, -0.0437372, 0.0233867, -0.0528773, 0.0670897

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0714282, upper bound: 0.0716281
time: 1.57 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733130, upper bound: 0.0721857
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0507102, 0.0096763, -0.0487823, 0.0096347, -0.0603449, 0.0584586
1: 0.9314473, 1.0253417, 0.9315191, 1.0207678, -0.0893205, 0.0938227
2: -0.0170213, 0.0465411, -0.0169818, 0.0434946, -0.0605159, 0.0635230
3: -0.0348589, 0.0082444, -0.0331020, 0.0082104, -0.0430693, 0.0413464
4: -0.0364481, 0.0233845, -0.0341849, 0.0232785, -0.0597266, 0.0575694
5: -0.0067404, 0.0629642, -0.0068704, 0.0599902, -0.0667306, 0.0698345
6: -0.0106630, 0.0216101, -0.0105842, 0.0218089, -0.0324719, 0.0321943
7: -0.0114996, 0.0881484, -0.0114652, 0.0841483, -0.0956479, 0.0996136
8: -0.0090915, 0.0326553, -0.0077873, 0.0326447, -0.0417362, 0.0404426
9: -0.0525798, 0.0234974, -0.0485412, 0.0234303, -0.0760101, 0.0720385

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0740026, upper bound: 0.0732979
time: 1.72 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0742823, upper bound: 0.0750428
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0394707, 0.0095301, -0.0505642, 0.0097246, -0.0491953, 0.0600943
1: 0.9315279, 1.0027279, 0.9313586, 1.0250183, -0.0934905, 0.0713693
2: -0.0169739, 0.0297220, -0.0170701, 0.0464689, -0.0634427, 0.0467921
3: -0.0247587, 0.0082047, -0.0347641, 0.0082865, -0.0330453, 0.0429688
4: -0.0228805, 0.0232616, -0.0363942, 0.0235155, -0.0463961, 0.0596559
5: -0.0057086, 0.0475331, -0.0069396, 0.0628194, -0.0685280, 0.0544727
6: -0.0105018, 0.0207460, -0.0107582, 0.0218340, -0.0323358, 0.0315042
7: -0.0113665, 0.0659306, -0.0115393, 0.0878555, -0.0992220, 0.0774699
8: -0.0070815, 0.0326434, -0.0091600, 0.0326684, -0.0397499, 0.0418034
9: -0.0311504, 0.0234222, -0.0522943, 0.0235803, -0.0547307, 0.0757165

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0721645, upper bound: 0.0718432
time: 1.62 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0742960, upper bound: 0.0723585
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0515498, 0.0097192, -0.0530477, 0.0097634, -0.0613133, 0.0627668
1: 0.9313717, 1.0273391, 0.9313115, 1.0308919, -0.0995201, 0.0960276
2: -0.0170629, 0.0478696, -0.0170963, 0.0502325, -0.0672953, 0.0649659
3: -0.0356297, 0.0082802, -0.0370045, 0.0083090, -0.0439387, 0.0452848
4: -0.0374945, 0.0234960, -0.0394429, 0.0235854, -0.0610799, 0.0629389
5: -0.0068287, 0.0641849, -0.0071892, 0.0662518, -0.0730805, 0.0713741
6: -0.0107453, 0.0216836, -0.0108183, 0.0220466, -0.0327919, 0.0325019
7: -0.0115350, 0.0898363, -0.0115728, 0.0927418, -0.1042768, 0.1014091
8: -0.0097660, 0.0326664, -0.0109685, 0.0326753, -0.0424413, 0.0436349
9: -0.0542278, 0.0235679, -0.0569951, 0.0236242, -0.0778521, 0.0805630

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734198, upper bound: 0.0748987
time: 1.36 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752204, upper bound: 0.0752484
time: 1.52 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0420447, 0.0094639, -0.0392446, 0.0095769, -0.0516216, 0.0487085
1: 0.9317365, 1.0055009, 0.9314365, 1.0033147, -0.0715783, 0.0740644
2: -0.0168608, 0.0332015, -0.0170239, 0.0293837, -0.0462445, 0.0502254
3: -0.0269968, 0.0081066, -0.0245802, 0.0082479, -0.0352446, 0.0326868
4: -0.0259314, 0.0229557, -0.0227823, 0.0233963, -0.0493277, 0.0457381
5: -0.0061975, 0.0505999, -0.0061754, 0.0470222, -0.0532197, 0.0567753
6: -0.0103128, 0.0212380, -0.0105976, 0.0212734, -0.0315862, 0.0318355
7: -0.0113183, 0.0707554, -0.0114045, 0.0653288, -0.0766471, 0.0821599
8: -0.0067532, 0.0326127, -0.0072409, 0.0326569, -0.0394101, 0.0398536
9: -0.0355860, 0.0232273, -0.0304053, 0.0235075, -0.0590935, 0.0536326

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0695632, upper bound: 0.0723351
time: 1.73 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0695632, upper bound: 0.0733051
time: 2.17 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0469387, 0.0095982, -0.0376706, 0.0094792, -0.0564179, 0.0472687
1: 0.9315633, 1.0164247, 0.9315825, 1.0030050, -0.0714417, 0.0848421
2: -0.0169572, 0.0406919, -0.0169432, 0.0269368, -0.0438940, 0.0576351
3: -0.0314340, 0.0081893, -0.0231312, 0.0081784, -0.0396124, 0.0313205
4: -0.0319269, 0.0232128, -0.0208336, 0.0231801, -0.0551070, 0.0440464
5: -0.0066956, 0.0574375, -0.0060098, 0.0447655, -0.0514611, 0.0634473
6: -0.0105277, 0.0216598, -0.0104277, 0.0211351, -0.0316628, 0.0320874
7: -0.0114336, 0.0804924, -0.0113220, 0.0621349, -0.0735685, 0.0918145
8: -0.0070954, 0.0326382, -0.0069660, 0.0326354, -0.0397307, 0.0396042
9: -0.0450147, 0.0233891, -0.0273296, 0.0233711, -0.0683858, 0.0507187

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694720, upper bound: 0.0711134
time: 2.87 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694720, upper bound: 0.0739852
time: 1.61 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0487823, 0.0096347, -0.0470045, 0.0096570, -0.0584393, 0.0566392
1: 0.9315191, 1.0207678, 0.9314536, 1.0165949, -0.0850758, 0.0893142
2: -0.0169818, 0.0434946, -0.0170174, 0.0408539, -0.0578358, 0.0605120
3: -0.0331020, 0.0082104, -0.0315187, 0.0082412, -0.0413432, 0.0397291
4: -0.0341849, 0.0232785, -0.0320716, 0.0233747, -0.0575595, 0.0553501
5: -0.0068704, 0.0599902, -0.0068141, 0.0575600, -0.0644304, 0.0668043
6: -0.0105842, 0.0218089, -0.0106447, 0.0218023, -0.0323865, 0.0324536
7: -0.0114652, 0.0841483, -0.0114817, 0.0806263, -0.0920915, 0.0956301
8: -0.0077873, 0.0326447, -0.0072891, 0.0326544, -0.0404416, 0.0399339
9: -0.0485412, 0.0234303, -0.0451266, 0.0234916, -0.0720327, 0.0685569

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0746423, upper bound: 0.0727773
time: 2.60 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0746423, upper bound: 0.0742699
time: 1.74 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0465068, 0.0096101, -0.0400910, 0.0096262, -0.0561330, 0.0497010
1: 0.9315275, 1.0154289, 0.9313619, 1.0034695, -0.0719420, 0.0840670
2: -0.0169766, 0.0402181, -0.0170652, 0.0306998, -0.0476764, 0.0572833
3: -0.0310748, 0.0082061, -0.0253549, 0.0082834, -0.0393582, 0.0335611
4: -0.0313902, 0.0232654, -0.0238216, 0.0235067, -0.0548969, 0.0470870
5: -0.0065200, 0.0571032, -0.0062580, 0.0482436, -0.0547636, 0.0633612
6: -0.0105605, 0.0214769, -0.0106839, 0.0213434, -0.0319039, 0.0321608
7: -0.0114424, 0.0797774, -0.0114461, 0.0670362, -0.0784786, 0.0912235
8: -0.0071522, 0.0326435, -0.0073808, 0.0326679, -0.0398201, 0.0400243
9: -0.0444712, 0.0234225, -0.0320627, 0.0235772, -0.0680484, 0.0554852

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722725, upper bound: 0.0722726
time: 1.75 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722725, upper bound: 0.0742843
time: 1.63 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0390226, 0.0094813, -0.0503207, 0.0097816, -0.0488042, 0.0598019
1: 0.9316180, 1.0029131, 0.9312550, 1.0244215, -0.0928034, 0.0716581
2: -0.0169243, 0.0290006, -0.0171270, 0.0460456, -0.0629699, 0.0461276
3: -0.0243345, 0.0081619, -0.0345547, 0.0083355, -0.0326700, 0.0427166
4: -0.0223720, 0.0231286, -0.0361913, 0.0236683, -0.0460403, 0.0593198
5: -0.0058861, 0.0467967, -0.0071610, 0.0622962, -0.0681823, 0.0539577
6: -0.0104052, 0.0209537, -0.0108697, 0.0220965, -0.0325017, 0.0318234
7: -0.0113265, 0.0649034, -0.0115861, 0.0872800, -0.0986065, 0.0764895
8: -0.0069218, 0.0326301, -0.0091771, 0.0326836, -0.0396054, 0.0418073
9: -0.0301028, 0.0233379, -0.0516032, 0.0236770, -0.0537799, 0.0749411

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733815, upper bound: 0.0733815
time: 1.46 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733815, upper bound: 0.0734089
time: 1.45 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0489645, 0.0096526, -0.0521772, 0.0098160, -0.0587804, 0.0618298
1: 0.9314806, 1.0212166, 0.9312109, 1.0288180, -0.0973374, 0.0900057
2: -0.0170029, 0.0439358, -0.0171515, 0.0488709, -0.0658738, 0.0610873
3: -0.0332932, 0.0082286, -0.0362372, 0.0083566, -0.0416498, 0.0444659
4: -0.0343963, 0.0233353, -0.0384722, 0.0237337, -0.0581300, 0.0618074
5: -0.0067690, 0.0604921, -0.0073358, 0.0648806, -0.0716495, 0.0678279
6: -0.0106233, 0.0216900, -0.0109245, 0.0222444, -0.0328677, 0.0326145
7: -0.0114795, 0.0846274, -0.0116156, 0.0909573, -0.1024368, 0.0962431
8: -0.0079311, 0.0326504, -0.0105137, 0.0326901, -0.0406212, 0.0431641
9: -0.0491445, 0.0234663, -0.0551477, 0.0237182, -0.0728627, 0.0786140

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734089, upper bound: 0.0748647
time: 1.52 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734089, upper bound: 0.0752067
time: 2.20 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.61 seconds
NS_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0715093, upper bound: 0.0716613
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0733645, upper bound: 0.0722038
NS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0740396, upper bound: 0.0733273
NS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0743315, upper bound: 0.0750819
NS_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0722271, upper bound: 0.0718723
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0743420, upper bound: 0.0723735
NS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0734754, upper bound: 0.0749455
NS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0752804, upper bound: 0.0752804
NS_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0714282, upper bound: 0.0716281
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0733130, upper bound: 0.0721857
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0740026, upper bound: 0.0732979
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0742823, upper bound: 0.0750428
NS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0721645, upper bound: 0.0718432
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0742960, upper bound: 0.0723585
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0734198, upper bound: 0.0748987
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0752204, upper bound: 0.0752484
NS_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0695632, upper bound: 0.0723351
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0695632, upper bound: 0.0733051
NS_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0694720, upper bound: 0.0711134
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0694720, upper bound: 0.0739852
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0746423, upper bound: 0.0727773
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0746423, upper bound: 0.0742699
NS_A2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0722725, upper bound: 0.0722726
NS_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0722725, upper bound: 0.0742843
NS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0733815, upper bound: 0.0733815
NS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0733815, upper bound: 0.0734089
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0734089, upper bound: 0.0748647
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.61
Output dim: 1, lower bound: -0.0734089, upper bound: 0.0752067

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0386245, 0.0094811, -0.0411425, 0.0092522, -0.0478767, 0.0506235
1: 0.9316024, 1.0025718, 0.9321389, 1.0038439, -0.0722415, 0.0704329
2: -0.0169327, 0.0284094, -0.0166400, 0.0315969, -0.0485295, 0.0450494
3: -0.0239849, 0.0081692, -0.0261185, 0.0079161, -0.0319010, 0.0342878
4: -0.0218442, 0.0231513, -0.0246482, 0.0223623, -0.0442065, 0.0477995
5: -0.0056231, 0.0463104, -0.0058510, 0.0492376, -0.0548607, 0.0521614
6: -0.0104157, 0.0206736, -0.0098867, 0.0208891, -0.0313047, 0.0305603
7: -0.0113252, 0.0642233, -0.0111456, 0.0689439, -0.0802691, 0.0753688
8: -0.0069419, 0.0326324, -0.0060461, 0.0325534, -0.0394954, 0.0386785
9: -0.0294906, 0.0233525, -0.0338460, 0.0228514, -0.0523420, 0.0571985

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715466, upper bound: 0.0721065
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715466, upper bound: 0.0722038
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0461104, 0.0093895, -0.0464329, 0.0554955
1: 0.9317459, 1.0022494, 0.9319614, 1.0144449, -0.0826991, 0.0702880
2: -0.0168533, 0.0259523, -0.0167388, 0.0391972, -0.0560505, 0.0426912
3: -0.0225295, 0.0081010, -0.0306241, 0.0080008, -0.0305303, 0.0387250
4: -0.0198907, 0.0229389, -0.0307383, 0.0226258, -0.0425165, 0.0536772
5: -0.0054429, 0.0440414, -0.0063526, 0.0561662, -0.0616091, 0.0503940
6: -0.0102488, 0.0205214, -0.0101068, 0.0213162, -0.0315649, 0.0306281
7: -0.0112441, 0.0610160, -0.0112634, 0.0788267, -0.0900709, 0.0722794
8: -0.0066718, 0.0326113, -0.0063965, 0.0325796, -0.0392514, 0.0390079
9: -0.0264003, 0.0232185, -0.0434040, 0.0230172, -0.0494175, 0.0666226

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0711595, upper bound: 0.0719522
time: 1.81 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0711595, upper bound: 0.0733274
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0479483, 0.0094258, -0.0557952, 0.0575092
1: 0.9316154, 1.0150608, 0.9319175, 1.0187615, -0.0871461, 0.0831432
2: -0.0169283, 0.0398458, -0.0167633, 0.0419881, -0.0589164, 0.0566090
3: -0.0309045, 0.0081644, -0.0322861, 0.0080218, -0.0389263, 0.0404505
4: -0.0310859, 0.0231355, -0.0329895, 0.0226909, -0.0537768, 0.0561250
5: -0.0062871, 0.0568534, -0.0065266, 0.0587051, -0.0649922, 0.0633799
6: -0.0104653, 0.0212261, -0.0101628, 0.0214652, -0.0319304, 0.0313889
7: -0.0114019, 0.0794904, -0.0112948, 0.0824653, -0.0938672, 0.0907852
8: -0.0069951, 0.0326305, -0.0066862, 0.0325860, -0.0395811, 0.0393168
9: -0.0441951, 0.0233403, -0.0469109, 0.0230581, -0.0672531, 0.0702513

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716236, upper bound: 0.0740523
time: 2.23 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716236, upper bound: 0.0750819
time: 1.44 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0394707, 0.0095301, -0.0456161, 0.0094182, -0.0488889, 0.0551462
1: 0.9315279, 1.0027279, 0.9318914, 1.0133041, -0.0817763, 0.0708364
2: -0.0169739, 0.0297220, -0.0167769, 0.0385886, -0.0555625, 0.0464989
3: -0.0247587, 0.0082047, -0.0302052, 0.0080338, -0.0327926, 0.0384099
4: -0.0228805, 0.0232616, -0.0301300, 0.0227286, -0.0456091, 0.0533916
5: -0.0057086, 0.0475331, -0.0061901, 0.0556896, -0.0613982, 0.0537232
6: -0.0105018, 0.0207460, -0.0101749, 0.0211494, -0.0316511, 0.0309208
7: -0.0113665, 0.0659306, -0.0112858, 0.0779768, -0.0893433, 0.0772164
8: -0.0070815, 0.0326434, -0.0065123, 0.0325899, -0.0396714, 0.0391557
9: -0.0311504, 0.0234222, -0.0426930, 0.0230825, -0.0542329, 0.0661152

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722959, upper bound: 0.0722959
time: 2.39 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722959, upper bound: 0.0723735
time: 1.98 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0496936, 0.0096851, -0.0381902, 0.0092888, -0.0589824, 0.0478753
1: 0.9314160, 1.0229414, 0.9319821, 1.0024300, -0.0710139, 0.0909593
2: -0.0170382, 0.0450448, -0.0167246, 0.0275023, -0.0445406, 0.0617694
3: -0.0339476, 0.0082591, -0.0235190, 0.0079896, -0.0419372, 0.0317781
4: -0.0352152, 0.0234303, -0.0211814, 0.0225917, -0.0578069, 0.0446116
5: -0.0066469, 0.0616019, -0.0055594, 0.0455038, -0.0521507, 0.0671612
6: -0.0106905, 0.0215297, -0.0100191, 0.0206247, -0.0313152, 0.0315487
7: -0.0115058, 0.0861613, -0.0111693, 0.0632219, -0.0747277, 0.0973305
8: -0.0084518, 0.0326599, -0.0062813, 0.0325765, -0.0410283, 0.0389411
9: -0.0506843, 0.0235265, -0.0284906, 0.0229979, -0.0736821, 0.0520171

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734531, upper bound: 0.0734531
time: 1.49 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734531, upper bound: 0.0749456
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0515498, 0.0097192, -0.0480822, 0.0094617, -0.0610115, 0.0578014
1: 0.9313717, 1.0273391, 0.9318430, 1.0190955, -0.0877238, 0.0954961
2: -0.0170629, 0.0478696, -0.0168041, 0.0423192, -0.0593821, 0.0646737
3: -0.0356297, 0.0082802, -0.0324318, 0.0080570, -0.0436867, 0.0407120
4: -0.0374945, 0.0234960, -0.0331458, 0.0228008, -0.0602952, 0.0566418
5: -0.0068287, 0.0641849, -0.0064393, 0.0590812, -0.0659099, 0.0706241
6: -0.0107453, 0.0216836, -0.0102394, 0.0213623, -0.0321075, 0.0319229
7: -0.0115350, 0.0898363, -0.0113237, 0.0828392, -0.0943742, 0.1011600
8: -0.0097660, 0.0326664, -0.0068562, 0.0325970, -0.0423630, 0.0395226
9: -0.0542278, 0.0235679, -0.0473738, 0.0231278, -0.0773556, 0.0709417

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749455, upper bound: 0.0734754
time: 1.41 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749455, upper bound: 0.0752804
time: 2.05 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0386245, 0.0094811, -0.0420447, 0.0094639, -0.0480884, 0.0515258
1: 0.9316024, 1.0025718, 0.9317365, 1.0055009, -0.0738985, 0.0708354
2: -0.0169327, 0.0284094, -0.0168608, 0.0332015, -0.0501341, 0.0452702
3: -0.0239849, 0.0081692, -0.0269968, 0.0081066, -0.0320915, 0.0351660
4: -0.0218442, 0.0231513, -0.0259314, 0.0229557, -0.0448000, 0.0490827
5: -0.0056231, 0.0463104, -0.0061975, 0.0505999, -0.0562230, 0.0525080
6: -0.0104157, 0.0206736, -0.0103128, 0.0212380, -0.0316536, 0.0309864
7: -0.0113252, 0.0642233, -0.0113183, 0.0707554, -0.0820806, 0.0755416
8: -0.0069419, 0.0326324, -0.0067532, 0.0326127, -0.0395546, 0.0393856
9: -0.0294906, 0.0233525, -0.0355860, 0.0232273, -0.0527179, 0.0589385

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723485, upper bound: 0.0696354
time: 4.31 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723485, upper bound: 0.0696354
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0469387, 0.0095982, -0.0466415, 0.0563238
1: 0.9317459, 1.0022494, 0.9315633, 1.0164247, -0.0846788, 0.0706860
2: -0.0168533, 0.0259523, -0.0169572, 0.0406919, -0.0575452, 0.0429095
3: -0.0225295, 0.0081010, -0.0314340, 0.0081893, -0.0307188, 0.0395350
4: -0.0198907, 0.0229389, -0.0319269, 0.0232128, -0.0431035, 0.0548658
5: -0.0054429, 0.0440414, -0.0066956, 0.0574375, -0.0628804, 0.0507371
6: -0.0102488, 0.0205214, -0.0105277, 0.0216598, -0.0319085, 0.0310491
7: -0.0112441, 0.0610160, -0.0114336, 0.0804924, -0.0917366, 0.0724496
8: -0.0066718, 0.0326113, -0.0070954, 0.0326382, -0.0393100, 0.0397067
9: -0.0264003, 0.0232185, -0.0450147, 0.0233891, -0.0497893, 0.0682332

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0711308, upper bound: 0.0718971
time: 1.60 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0711308, upper bound: 0.0732979
time: 1.82 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0487823, 0.0096347, -0.0560042, 0.0583432
1: 0.9316154, 1.0150608, 0.9315191, 1.0207678, -0.0891524, 0.0835417
2: -0.0169283, 0.0398458, -0.0169818, 0.0434946, -0.0604228, 0.0568276
3: -0.0309045, 0.0081644, -0.0331020, 0.0082104, -0.0391149, 0.0412664
4: -0.0310859, 0.0231355, -0.0341849, 0.0232785, -0.0543645, 0.0573204
5: -0.0062871, 0.0568534, -0.0068704, 0.0599902, -0.0662773, 0.0637237
6: -0.0104653, 0.0212261, -0.0105842, 0.0218089, -0.0322742, 0.0318103
7: -0.0114019, 0.0794904, -0.0114652, 0.0841483, -0.0955502, 0.0909556
8: -0.0069951, 0.0326305, -0.0077873, 0.0326447, -0.0396398, 0.0404178
9: -0.0441951, 0.0233403, -0.0485412, 0.0234303, -0.0676254, 0.0718815

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0715922, upper bound: 0.0740142
time: 2.36 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0715922, upper bound: 0.0750428
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0394707, 0.0095301, -0.0465068, 0.0096101, -0.0490807, 0.0560369
1: 0.9315279, 1.0027279, 0.9315275, 1.0154289, -0.0839010, 0.0712004
2: -0.0169739, 0.0297220, -0.0169766, 0.0402181, -0.0571920, 0.0466986
3: -0.0247587, 0.0082047, -0.0310748, 0.0082061, -0.0329649, 0.0392795
4: -0.0228805, 0.0232616, -0.0313902, 0.0232654, -0.0461459, 0.0546518
5: -0.0057086, 0.0475331, -0.0065200, 0.0571032, -0.0628118, 0.0540531
6: -0.0105018, 0.0207460, -0.0105605, 0.0214769, -0.0319787, 0.0313065
7: -0.0113665, 0.0659306, -0.0114424, 0.0797774, -0.0911439, 0.0773730
8: -0.0070815, 0.0326434, -0.0071522, 0.0326435, -0.0397250, 0.0397956
9: -0.0311504, 0.0234222, -0.0444712, 0.0234225, -0.0545729, 0.0678934

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722880, upper bound: 0.0722746
time: 1.54 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722880, upper bound: 0.0723585
time: 2.44 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0496936, 0.0096851, -0.0390226, 0.0094813, -0.0591749, 0.0487077
1: 0.9314160, 1.0229414, 0.9316180, 1.0029131, -0.0714971, 0.0913233
2: -0.0170382, 0.0450448, -0.0169243, 0.0290006, -0.0460389, 0.0619691
3: -0.0339476, 0.0082591, -0.0243345, 0.0081619, -0.0421096, 0.0325936
4: -0.0352152, 0.0234303, -0.0223720, 0.0231286, -0.0583438, 0.0458022
5: -0.0066469, 0.0616019, -0.0058861, 0.0467967, -0.0534437, 0.0674880
6: -0.0106905, 0.0215297, -0.0104052, 0.0209537, -0.0316442, 0.0319349
7: -0.0115058, 0.0861613, -0.0113265, 0.0649034, -0.0764092, 0.0974878
8: -0.0084518, 0.0326599, -0.0069218, 0.0326301, -0.0410819, 0.0395817
9: -0.0506843, 0.0235265, -0.0301028, 0.0233379, -0.0740222, 0.0536294

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733957, upper bound: 0.0734218
time: 1.75 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733957, upper bound: 0.0748987
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0515498, 0.0097192, -0.0489645, 0.0096526, -0.0612025, 0.0586836
1: 0.9313717, 1.0273391, 0.9314806, 1.0212166, -0.0898449, 0.0958585
2: -0.0170629, 0.0478696, -0.0170029, 0.0439358, -0.0609986, 0.0648725
3: -0.0356297, 0.0082802, -0.0332932, 0.0082286, -0.0438583, 0.0415734
4: -0.0374945, 0.0234960, -0.0343963, 0.0233353, -0.0608297, 0.0578923
5: -0.0068287, 0.0641849, -0.0067690, 0.0604921, -0.0673208, 0.0709538
6: -0.0107453, 0.0216836, -0.0106233, 0.0216900, -0.0324352, 0.0323069
7: -0.0115350, 0.0898363, -0.0114795, 0.0846274, -0.0961625, 0.1013158
8: -0.0097660, 0.0326664, -0.0079311, 0.0326504, -0.0424164, 0.0405975
9: -0.0542278, 0.0235679, -0.0491445, 0.0234663, -0.0776942, 0.0727124

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748864, upper bound: 0.0734507
time: 1.65 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748864, upper bound: 0.0752484
time: 1.81 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0420447, 0.0094639, -0.0353997, 0.0094241, -0.0514688, 0.0448637
1: 0.9317365, 1.0055009, 0.9316030, 1.0025769, -0.0708405, 0.0738979
2: -0.0168608, 0.0332015, -0.0169305, 0.0236061, -0.0404670, 0.0501320
3: -0.0269968, 0.0081066, -0.0211019, 0.0081680, -0.0351647, 0.0292085
4: -0.0259314, 0.0229557, -0.0181529, 0.0231481, -0.0490795, 0.0411086
5: -0.0061975, 0.0505999, -0.0058058, 0.0416564, -0.0478539, 0.0564056
6: -0.0103128, 0.0212380, -0.0103735, 0.0209606, -0.0312733, 0.0316115
7: -0.0113183, 0.0707554, -0.0112711, 0.0576449, -0.0689632, 0.0820265
8: -0.0067532, 0.0326127, -0.0068911, 0.0326324, -0.0393855, 0.0395038
9: -0.0355860, 0.0232273, -0.0230350, 0.0233519, -0.0589380, 0.0462623

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694719, upper bound: 0.0715018
time: 1.40 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694719, upper bound: 0.0708970
time: 3.86 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0434649, 0.0095334, -0.0376706, 0.0094792, -0.0529441, 0.0472040
1: 0.9316280, 1.0084171, 0.9315825, 1.0030050, -0.0713770, 0.0768346
2: -0.0169207, 0.0354182, -0.0169432, 0.0269368, -0.0438574, 0.0523614
3: -0.0282976, 0.0081581, -0.0231312, 0.0081784, -0.0364760, 0.0312893
4: -0.0276867, 0.0231160, -0.0208336, 0.0231801, -0.0508667, 0.0439496
5: -0.0063406, 0.0526322, -0.0060098, 0.0447655, -0.0511061, 0.0586421
6: -0.0104366, 0.0213583, -0.0104277, 0.0211351, -0.0315717, 0.0317860
7: -0.0113766, 0.0736180, -0.0113220, 0.0621349, -0.0735116, 0.0849401
8: -0.0069546, 0.0326287, -0.0069660, 0.0326354, -0.0395899, 0.0395947
9: -0.0383706, 0.0233285, -0.0273296, 0.0233711, -0.0617417, 0.0506581

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0727499
time: 1.39 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0739852
time: 1.87 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0350170, 0.0093074, -0.0470045, 0.0096570, -0.0446740, 0.0563119
1: 0.9318230, 1.0023025, 0.9314536, 1.0165949, -0.0847719, 0.0708489
2: -0.0168098, 0.0228257, -0.0170174, 0.0408539, -0.0576637, 0.0398431
3: -0.0206715, 0.0080639, -0.0315187, 0.0082412, -0.0289127, 0.0395826
4: -0.0175318, 0.0228237, -0.0320716, 0.0233747, -0.0409064, 0.0548953
5: -0.0056287, 0.0409862, -0.0068141, 0.0575600, -0.0631887, 0.0478003
6: -0.0101399, 0.0207694, -0.0106447, 0.0218023, -0.0319422, 0.0314141
7: -0.0111758, 0.0567504, -0.0114817, 0.0806263, -0.0918021, 0.0682321
8: -0.0065038, 0.0326000, -0.0072891, 0.0326544, -0.0391581, 0.0398891
9: -0.0221895, 0.0231465, -0.0451266, 0.0234916, -0.0456811, 0.0682731

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0674237, upper bound: 0.0692479
time: 1.63 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730042, upper bound: 0.0725109
time: 2.01 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0444981, 0.0095132, -0.0470045, 0.0096570, -0.0541551, 0.0565177
1: 0.9316897, 1.0107032, 0.9314536, 1.0165949, -0.0849051, 0.0792496
2: -0.0168872, 0.0368815, -0.0170174, 0.0408539, -0.0577411, 0.0538989
3: -0.0291996, 0.0081291, -0.0315187, 0.0082412, -0.0374408, 0.0396478
4: -0.0289021, 0.0230256, -0.0320716, 0.0233747, -0.0522767, 0.0550971
5: -0.0064369, 0.0539610, -0.0068141, 0.0575600, -0.0639970, 0.0607751
6: -0.0103803, 0.0214412, -0.0106447, 0.0218023, -0.0321826, 0.0320858
7: -0.0113618, 0.0755931, -0.0114817, 0.0806263, -0.0919881, 0.0870749
8: -0.0068570, 0.0326196, -0.0072891, 0.0326544, -0.0395113, 0.0399087
9: -0.0402459, 0.0232709, -0.0451266, 0.0234916, -0.0637375, 0.0683975

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0715829
time: 1.64 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0742699
time: 1.44 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0451680, 0.0095883, -0.0400910, 0.0096262, -0.0547942, 0.0496793
1: 0.9315471, 1.0122955, 0.9313619, 1.0034695, -0.0719224, 0.0809336
2: -0.0169654, 0.0381836, -0.0170652, 0.0306998, -0.0476652, 0.0552487
3: -0.0298564, 0.0081966, -0.0253549, 0.0082834, -0.0381397, 0.0335515
4: -0.0297433, 0.0232359, -0.0238216, 0.0235067, -0.0532500, 0.0470575
5: -0.0063871, 0.0552426, -0.0062580, 0.0482436, -0.0546306, 0.0615006
6: -0.0105313, 0.0213638, -0.0106839, 0.0213434, -0.0318747, 0.0320477
7: -0.0114231, 0.0770954, -0.0114461, 0.0670362, -0.0784593, 0.0885414
8: -0.0071076, 0.0326406, -0.0073808, 0.0326679, -0.0397754, 0.0400214
9: -0.0418954, 0.0234041, -0.0320627, 0.0235772, -0.0654726, 0.0554668

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0732780
time: 2.03 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0742843
time: 1.69 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0390226, 0.0094813, -0.0384924, 0.0095304, -0.0485530, 0.0479736
1: 0.9316180, 1.0029131, 0.9315068, 1.0031613, -0.0715433, 0.0714063
2: -0.0169243, 0.0290006, -0.0169850, 0.0282126, -0.0451369, 0.0459857
3: -0.0243345, 0.0081619, -0.0238848, 0.0082144, -0.0325489, 0.0320467
4: -0.0223720, 0.0231286, -0.0218421, 0.0232922, -0.0456642, 0.0449706
5: -0.0058861, 0.0467967, -0.0060880, 0.0459481, -0.0518342, 0.0528847
6: -0.0104052, 0.0209537, -0.0105161, 0.0212012, -0.0316064, 0.0314698
7: -0.0113265, 0.0649034, -0.0113653, 0.0637961, -0.0751226, 0.0762687
8: -0.0069218, 0.0326301, -0.0071090, 0.0326465, -0.0395683, 0.0397391
9: -0.0301028, 0.0233379, -0.0289352, 0.0234418, -0.0535446, 0.0522731

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0695965, upper bound: 0.0696899
time: 1.53 seconds

## Relational analysis of NS_A2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0695965, upper bound: 0.0733815
time: 2.33 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0390226, 0.0094813, -0.0478514, 0.0097020, -0.0487246, 0.0573327
1: 0.9316180, 1.0029131, 0.9313772, 1.0185810, -0.0869630, 0.0715359
2: -0.0169243, 0.0290006, -0.0170595, 0.0421917, -0.0591161, 0.0460601
3: -0.0243345, 0.0081619, -0.0322955, 0.0082775, -0.0326119, 0.0404575
4: -0.0223720, 0.0231286, -0.0331239, 0.0234875, -0.0458595, 0.0562524
5: -0.0058861, 0.0467967, -0.0068993, 0.0587832, -0.0646693, 0.0536961
6: -0.0104052, 0.0209537, -0.0107291, 0.0218741, -0.0322793, 0.0316828
7: -0.0113265, 0.0649034, -0.0115191, 0.0823282, -0.0936547, 0.0764225
8: -0.0069218, 0.0326301, -0.0074276, 0.0326656, -0.0395874, 0.0400577
9: -0.0301028, 0.0233379, -0.0467909, 0.0235629, -0.0536657, 0.0701288

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0675798, upper bound: 0.0700396
time: 1.41 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731406, upper bound: 0.0731677
time: 2.23 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0489645, 0.0096526, -0.0384924, 0.0095304, -0.0584949, 0.0481450
1: 0.9314806, 1.0212166, 0.9315068, 1.0031613, -0.0716807, 0.0897098
2: -0.0170029, 0.0439358, -0.0169850, 0.0282126, -0.0452155, 0.0609208
3: -0.0332932, 0.0082286, -0.0238848, 0.0082144, -0.0415076, 0.0321134
4: -0.0343963, 0.0233353, -0.0218421, 0.0232922, -0.0576885, 0.0451773
5: -0.0067690, 0.0604921, -0.0060880, 0.0459481, -0.0527171, 0.0665801
6: -0.0106233, 0.0216900, -0.0105161, 0.0212012, -0.0318245, 0.0322061
7: -0.0114795, 0.0846274, -0.0113653, 0.0637961, -0.0752756, 0.0959927
8: -0.0079311, 0.0326504, -0.0071090, 0.0326465, -0.0405776, 0.0397594
9: -0.0491445, 0.0234663, -0.0289352, 0.0234418, -0.0725863, 0.0524015

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0718406
time: 2.03 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0748647
time: 1.46 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0489645, 0.0096526, -0.0478514, 0.0097020, -0.0586665, 0.0575041
1: 0.9314806, 1.0212166, 0.9313772, 1.0185810, -0.0871004, 0.0898394
2: -0.0170029, 0.0439358, -0.0170595, 0.0421917, -0.0591947, 0.0609952
3: -0.0332932, 0.0082286, -0.0322955, 0.0082775, -0.0415707, 0.0405242
4: -0.0343963, 0.0233353, -0.0331239, 0.0234875, -0.0578838, 0.0564592
5: -0.0067690, 0.0604921, -0.0068993, 0.0587832, -0.0655522, 0.0673914
6: -0.0106233, 0.0216900, -0.0107291, 0.0218741, -0.0324974, 0.0324191
7: -0.0114795, 0.0846274, -0.0115191, 0.0823282, -0.0938077, 0.0961466
8: -0.0079311, 0.0326504, -0.0074276, 0.0326656, -0.0405967, 0.0400779
9: -0.0491445, 0.0234663, -0.0467909, 0.0235629, -0.0727074, 0.0702572

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0723536
time: 2.53 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0752067
time: 1.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.13 seconds
NS_A1_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0715466, upper bound: 0.0721065
NS_A1_B1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0715466, upper bound: 0.0722038
NS_A1_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0711595, upper bound: 0.0719522
NS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0711595, upper bound: 0.0733274
NS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0716236, upper bound: 0.0740523
NS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0716236, upper bound: 0.0750819
NS_A1_B1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0722959, upper bound: 0.0722959
NS_A1_B1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0722959, upper bound: 0.0723735
NS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0734531, upper bound: 0.0734531
NS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0734531, upper bound: 0.0749456
NS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0749455, upper bound: 0.0734754
NS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0749455, upper bound: 0.0752804
NS_A1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0723485, upper bound: 0.0696354
NS_A1_B2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0723485, upper bound: 0.0696354
NS_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0711308, upper bound: 0.0718971
NS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0711308, upper bound: 0.0732979
NS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0715922, upper bound: 0.0740142
NS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0715922, upper bound: 0.0750428
NS_A1_B2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0722880, upper bound: 0.0722746
NS_A1_B2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0722880, upper bound: 0.0723585
NS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0733957, upper bound: 0.0734218
NS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0733957, upper bound: 0.0748987
NS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0748864, upper bound: 0.0734507
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0748864, upper bound: 0.0752484
NS_A2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0694719, upper bound: 0.0715018
NS_A2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0694719, upper bound: 0.0708970
NS_A2_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0727499
NS_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0739852
NS_A2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0674237, upper bound: 0.0692479
NS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0730042, upper bound: 0.0725109
NS_A2_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0715829
NS_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0694390, upper bound: 0.0742699
NS_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0732780
NS_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0742843
NS_A2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0695965, upper bound: 0.0696899
NS_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0695965, upper bound: 0.0733815
NS_A2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0675798, upper bound: 0.0700396
NS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0731406, upper bound: 0.0731677
NS_A2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0718406
NS_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0748647
NS_A2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0723536
NS_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.13
Output dim: 1, lower bound: -0.0696430, upper bound: 0.0752067

## BFS NS instance: NS_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0426249, 0.0093243, -0.0463677, 0.0520100
1: 0.9317459, 1.0022494, 0.9320268, 1.0065711, -0.0748252, 0.0702226
2: -0.0168533, 0.0259523, -0.0167020, 0.0339069, -0.0507602, 0.0426543
3: -0.0225295, 0.0081010, -0.0274786, 0.0079694, -0.0304989, 0.0355796
4: -0.0198907, 0.0229389, -0.0264817, 0.0225281, -0.0424188, 0.0494207
5: -0.0054429, 0.0440414, -0.0059984, 0.0513433, -0.0567862, 0.0500398
6: -0.0102488, 0.0205214, -0.0100150, 0.0210138, -0.0312625, 0.0305364
7: -0.0112441, 0.0610160, -0.0112062, 0.0719347, -0.0831788, 0.0722221
8: -0.0066718, 0.0326113, -0.0062546, 0.0325699, -0.0392418, 0.0388659
9: -0.0264003, 0.0232185, -0.0367494, 0.0229561, -0.0493564, 0.0599679

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691779, upper bound: 0.0733119
time: 1.53 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691779, upper bound: 0.0733273
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0323475, 0.0090902, -0.0554597, 0.0419084
1: 0.9316154, 1.0150608, 0.9321362, 1.0013850, -0.0697696, 0.0829246
2: -0.0169283, 0.0398458, -0.0166364, 0.0187601, -0.0356883, 0.0564821
3: -0.0309045, 0.0081644, -0.0182519, 0.0079148, -0.0388192, 0.0264163
4: -0.0310859, 0.0231355, -0.0142305, 0.0223598, -0.0534457, 0.0373660
5: -0.0062871, 0.0568534, -0.0051064, 0.0373466, -0.0436337, 0.0619598
6: -0.0104653, 0.0212261, -0.0097700, 0.0202635, -0.0307288, 0.0309961
7: -0.0114019, 0.0794904, -0.0109916, 0.0515266, -0.0629284, 0.0904820
8: -0.0069951, 0.0326305, -0.0059074, 0.0325539, -0.0395489, 0.0385379
9: -0.0441951, 0.0233403, -0.0172842, 0.0228539, -0.0670490, 0.0406245

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0673635, upper bound: 0.0673628
time: 1.48 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714235, upper bound: 0.0738269
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0444551, 0.0093644, -0.0557338, 0.0540160
1: 0.9316154, 1.0150608, 0.9319829, 1.0106126, -0.0789972, 0.0830779
2: -0.0169283, 0.0398458, -0.0167266, 0.0366787, -0.0536069, 0.0565723
3: -0.0309045, 0.0081644, -0.0291335, 0.0079904, -0.0388949, 0.0372979
4: -0.0310859, 0.0231355, -0.0287121, 0.0225935, -0.0536794, 0.0518476
5: -0.0062871, 0.0568534, -0.0061708, 0.0538703, -0.0601574, 0.0630241
6: -0.0104653, 0.0212261, -0.0100738, 0.0211600, -0.0316253, 0.0312999
7: -0.0114019, 0.0794904, -0.0112411, 0.0755709, -0.0869727, 0.0907315
8: -0.0069951, 0.0326305, -0.0063465, 0.0325764, -0.0395715, 0.0389771
9: -0.0441951, 0.0233403, -0.0402573, 0.0229971, -0.0671921, 0.0635976

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0692037, upper bound: 0.0747204
time: 1.58 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0692037, upper bound: 0.0747204
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0381902, 0.0092888, -0.0471542, 0.0476261
1: 0.9316702, 1.0024072, 0.9319821, 1.0024300, -0.0707598, 0.0704251
2: -0.0168951, 0.0272271, -0.0167246, 0.0275023, -0.0443975, 0.0439517
3: -0.0232820, 0.0081369, -0.0235190, 0.0079896, -0.0312716, 0.0316559
4: -0.0208966, 0.0230509, -0.0211814, 0.0225917, -0.0434883, 0.0442322
5: -0.0055243, 0.0452254, -0.0055594, 0.0455038, -0.0510280, 0.0507847
6: -0.0103369, 0.0205902, -0.0100191, 0.0206247, -0.0309616, 0.0306093
7: -0.0112871, 0.0626766, -0.0111693, 0.0632219, -0.0745090, 0.0738459
8: -0.0068144, 0.0326225, -0.0062813, 0.0325765, -0.0393910, 0.0389037
9: -0.0280082, 0.0232892, -0.0284906, 0.0229979, -0.0510061, 0.0517798

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0720837
time: 4.73 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0734531
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0381902, 0.0092888, -0.0565042, 0.0477956
1: 0.9315395, 1.0170586, 0.9319821, 1.0024300, -0.0708904, 0.0850765
2: -0.0169700, 0.0411812, -0.0167246, 0.0275023, -0.0444723, 0.0579058
3: -0.0316799, 0.0082004, -0.0235190, 0.0079896, -0.0396695, 0.0317194
4: -0.0321350, 0.0232476, -0.0211814, 0.0225917, -0.0547267, 0.0444290
5: -0.0063744, 0.0580774, -0.0055594, 0.0455038, -0.0518781, 0.0636368
6: -0.0105490, 0.0212991, -0.0100191, 0.0206247, -0.0311737, 0.0313182
7: -0.0114388, 0.0811907, -0.0111693, 0.0632219, -0.0746607, 0.0923600
8: -0.0071325, 0.0326417, -0.0062813, 0.0325765, -0.0397090, 0.0389230
9: -0.0458587, 0.0234112, -0.0284906, 0.0229979, -0.0688566, 0.0519018

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698643, upper bound: 0.0694014
time: 1.88 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732190, upper bound: 0.0747076
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0480822, 0.0094617, -0.0473271, 0.0575182
1: 0.9316702, 1.0024072, 0.9318430, 1.0190955, -0.0874254, 0.0705642
2: -0.0168951, 0.0272271, -0.0168041, 0.0423192, -0.0592144, 0.0440312
3: -0.0232820, 0.0081369, -0.0324318, 0.0080570, -0.0313391, 0.0405687
4: -0.0208966, 0.0230509, -0.0331458, 0.0228008, -0.0436974, 0.0561967
5: -0.0055243, 0.0452254, -0.0064393, 0.0590812, -0.0646055, 0.0516646
6: -0.0103369, 0.0205902, -0.0102394, 0.0213623, -0.0316992, 0.0308296
7: -0.0112871, 0.0626766, -0.0113237, 0.0828392, -0.0941263, 0.0740003
8: -0.0068144, 0.0326225, -0.0068562, 0.0325970, -0.0394114, 0.0394786
9: -0.0280082, 0.0232892, -0.0473738, 0.0231278, -0.0511360, 0.0706630

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0722271
time: 1.59 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0734754
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0480822, 0.0094617, -0.0566772, 0.0576876
1: 0.9315395, 1.0170586, 0.9318430, 1.0190955, -0.0875560, 0.0852156
2: -0.0169700, 0.0411812, -0.0168041, 0.0423192, -0.0592893, 0.0579853
3: -0.0316799, 0.0082004, -0.0324318, 0.0080570, -0.0397369, 0.0406322
4: -0.0321350, 0.0232476, -0.0331458, 0.0228008, -0.0549358, 0.0563934
5: -0.0063744, 0.0580774, -0.0064393, 0.0590812, -0.0654556, 0.0645167
6: -0.0105490, 0.0212991, -0.0102394, 0.0213623, -0.0319113, 0.0315385
7: -0.0114388, 0.0811907, -0.0113237, 0.0828392, -0.0942780, 0.0925144
8: -0.0071325, 0.0326417, -0.0068562, 0.0325970, -0.0397295, 0.0394979
9: -0.0458587, 0.0234112, -0.0473738, 0.0231278, -0.0689865, 0.0707850

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0743420
time: 1.39 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0752804
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0434649, 0.0095334, -0.0465767, 0.0528500
1: 0.9317459, 1.0022494, 0.9316280, 1.0084171, -0.0766712, 0.0706213
2: -0.0168533, 0.0259523, -0.0169207, 0.0354182, -0.0522715, 0.0428730
3: -0.0225295, 0.0081010, -0.0282976, 0.0081581, -0.0306876, 0.0363986
4: -0.0198907, 0.0229389, -0.0276867, 0.0231160, -0.0430067, 0.0506256
5: -0.0054429, 0.0440414, -0.0063406, 0.0526322, -0.0580751, 0.0503821
6: -0.0102488, 0.0205214, -0.0104366, 0.0213583, -0.0316071, 0.0309580
7: -0.0112441, 0.0610160, -0.0113766, 0.0736180, -0.0848622, 0.0723926
8: -0.0066718, 0.0326113, -0.0069546, 0.0326287, -0.0393005, 0.0395659
9: -0.0264003, 0.0232185, -0.0383706, 0.0233285, -0.0497288, 0.0615892

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691023, upper bound: 0.0732785
time: 1.96 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691023, upper bound: 0.0732979
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0333081, 0.0093131, -0.0556826, 0.0428690
1: 0.9316154, 1.0150608, 0.9317247, 1.0019433, -0.0703279, 0.0833361
2: -0.0169283, 0.0398458, -0.0168623, 0.0204524, -0.0373807, 0.0567080
3: -0.0309045, 0.0081644, -0.0191909, 0.0081097, -0.0390141, 0.0273553
4: -0.0310859, 0.0231355, -0.0155869, 0.0229668, -0.0540528, 0.0387224
5: -0.0062871, 0.0568534, -0.0054681, 0.0387902, -0.0450772, 0.0623215
6: -0.0104653, 0.0212261, -0.0102104, 0.0206286, -0.0310939, 0.0314365
7: -0.0114019, 0.0794904, -0.0111743, 0.0534653, -0.0648671, 0.0906647
8: -0.0069951, 0.0326305, -0.0066361, 0.0326144, -0.0396095, 0.0392666
9: -0.0441951, 0.0233403, -0.0190954, 0.0232383, -0.0674334, 0.0424358

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0673582, upper bound: 0.0673519
time: 2.46 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0713885, upper bound: 0.0737857
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0452795, 0.0095733, -0.0559428, 0.0548404
1: 0.9316154, 1.0150608, 0.9315839, 1.0125546, -0.0809392, 0.0834769
2: -0.0169283, 0.0398458, -0.0169454, 0.0381656, -0.0550938, 0.0567911
3: -0.0309045, 0.0081644, -0.0299352, 0.0081793, -0.0390837, 0.0380996
4: -0.0310859, 0.0231355, -0.0298928, 0.0231818, -0.0542677, 0.0530283
5: -0.0062871, 0.0568534, -0.0065108, 0.0551422, -0.0614293, 0.0633642
6: -0.0104653, 0.0212261, -0.0104956, 0.0215021, -0.0319674, 0.0317217
7: -0.0114019, 0.0794904, -0.0114115, 0.0772171, -0.0886189, 0.0909019
8: -0.0069951, 0.0326305, -0.0070468, 0.0326352, -0.0396302, 0.0396773
9: -0.0441951, 0.0233403, -0.0418499, 0.0233697, -0.0675648, 0.0651902

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691275, upper bound: 0.0746696
time: 1.54 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691275, upper bound: 0.0750428
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0390226, 0.0094813, -0.0473467, 0.0484586
1: 0.9316702, 1.0024072, 0.9316180, 1.0029131, -0.0712429, 0.0707892
2: -0.0168951, 0.0272271, -0.0169243, 0.0290006, -0.0458958, 0.0441514
3: -0.0232820, 0.0081369, -0.0243345, 0.0081619, -0.0314440, 0.0324714
4: -0.0208966, 0.0230509, -0.0223720, 0.0231286, -0.0440251, 0.0454229
5: -0.0055243, 0.0452254, -0.0058861, 0.0467967, -0.0523210, 0.0511115
6: -0.0103369, 0.0205902, -0.0104052, 0.0209537, -0.0312906, 0.0309954
7: -0.0112871, 0.0626766, -0.0113265, 0.0649034, -0.0761905, 0.0740031
8: -0.0068144, 0.0326225, -0.0069218, 0.0326301, -0.0394446, 0.0395443
9: -0.0280082, 0.0232892, -0.0301028, 0.0233379, -0.0513462, 0.0533920

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0720353
time: 1.41 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0734218
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0390226, 0.0094813, -0.0566967, 0.0486280
1: 0.9315395, 1.0170586, 0.9316180, 1.0029131, -0.0713736, 0.0854406
2: -0.0169700, 0.0411812, -0.0169243, 0.0290006, -0.0459707, 0.0581056
3: -0.0316799, 0.0082004, -0.0243345, 0.0081619, -0.0398418, 0.0325349
4: -0.0321350, 0.0232476, -0.0223720, 0.0231286, -0.0552636, 0.0456196
5: -0.0063744, 0.0580774, -0.0058861, 0.0467967, -0.0531711, 0.0639636
6: -0.0105490, 0.0212991, -0.0104052, 0.0209537, -0.0315027, 0.0317044
7: -0.0114388, 0.0811907, -0.0113265, 0.0649034, -0.0763422, 0.0925172
8: -0.0071325, 0.0326417, -0.0069218, 0.0326301, -0.0397626, 0.0395635
9: -0.0458587, 0.0234112, -0.0301028, 0.0233379, -0.0691966, 0.0535140

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698229, upper bound: 0.0693732
time: 2.15 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731545, upper bound: 0.0746605
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0489645, 0.0096526, -0.0475180, 0.0584004
1: 0.9316702, 1.0024072, 0.9314806, 1.0212166, -0.0895464, 0.0709266
2: -0.0168951, 0.0272271, -0.0170029, 0.0439358, -0.0608309, 0.0442300
3: -0.0232820, 0.0081369, -0.0332932, 0.0082286, -0.0315107, 0.0414301
4: -0.0208966, 0.0230509, -0.0343963, 0.0233353, -0.0442319, 0.0574472
5: -0.0055243, 0.0452254, -0.0067690, 0.0604921, -0.0660163, 0.0519943
6: -0.0103369, 0.0205902, -0.0106233, 0.0216900, -0.0320269, 0.0312135
7: -0.0112871, 0.0626766, -0.0114795, 0.0846274, -0.0959145, 0.0741561
8: -0.0068144, 0.0326225, -0.0079311, 0.0326504, -0.0394648, 0.0405535
9: -0.0280082, 0.0232892, -0.0491445, 0.0234663, -0.0514746, 0.0724337

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0721869
time: 1.43 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0734507
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0489645, 0.0096526, -0.0568681, 0.0585699
1: 0.9315395, 1.0170586, 0.9314806, 1.0212166, -0.0896771, 0.0855780
2: -0.0169700, 0.0411812, -0.0170029, 0.0439358, -0.0609058, 0.0581841
3: -0.0316799, 0.0082004, -0.0332932, 0.0082286, -0.0399085, 0.0414936
4: -0.0321350, 0.0232476, -0.0343963, 0.0233353, -0.0554703, 0.0576439
5: -0.0063744, 0.0580774, -0.0067690, 0.0604921, -0.0668664, 0.0648464
6: -0.0105490, 0.0212991, -0.0106233, 0.0216900, -0.0322390, 0.0319225
7: -0.0114388, 0.0811907, -0.0114795, 0.0846274, -0.0960663, 0.0926702
8: -0.0071325, 0.0326417, -0.0079311, 0.0326504, -0.0397829, 0.0405728
9: -0.0458587, 0.0234112, -0.0491445, 0.0234663, -0.0693251, 0.0725558

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0743108
time: 2.70 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0752484
time: 1.62 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0408737, 0.0094401, -0.0376706, 0.0094792, -0.0503529, 0.0471107
1: 0.9317551, 1.0036863, 0.9315825, 1.0030050, -0.0712500, 0.0721038
2: -0.0168501, 0.0314081, -0.0169432, 0.0269368, -0.0437869, 0.0483513
3: -0.0259266, 0.0080975, -0.0231312, 0.0081784, -0.0341049, 0.0312287
4: -0.0244918, 0.0229276, -0.0208336, 0.0231801, -0.0476719, 0.0437612
5: -0.0060774, 0.0489587, -0.0060098, 0.0447655, -0.0508429, 0.0549685
6: -0.0102828, 0.0211351, -0.0104277, 0.0211351, -0.0314179, 0.0315628
7: -0.0112970, 0.0683942, -0.0113220, 0.0621349, -0.0734319, 0.0797162
8: -0.0067081, 0.0326099, -0.0069660, 0.0326354, -0.0393435, 0.0395759
9: -0.0333032, 0.0232098, -0.0273296, 0.0233711, -0.0566743, 0.0505394

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692803, upper bound: 0.0684439
time: 1.48 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729591, upper bound: 0.0737441
time: 3.29 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0350170, 0.0093074, -0.0450081, 0.0096041, -0.0446211, 0.0543155
1: 0.9318230, 1.0023025, 0.9315236, 1.0119632, -0.0801403, 0.0707789
2: -0.0168098, 0.0228257, -0.0169785, 0.0378269, -0.0546367, 0.0398042
3: -0.0206715, 0.0080639, -0.0297116, 0.0082078, -0.0288793, 0.0377755
4: -0.0175318, 0.0228237, -0.0296265, 0.0232707, -0.0408025, 0.0524502
5: -0.0056287, 0.0409862, -0.0066148, 0.0548095, -0.0604382, 0.0476010
6: -0.0101399, 0.0207694, -0.0105587, 0.0216345, -0.0317744, 0.0313281
7: -0.0111758, 0.0567504, -0.0114364, 0.0766620, -0.0878378, 0.0681868
8: -0.0065038, 0.0326000, -0.0071519, 0.0326440, -0.0391478, 0.0397519
9: -0.0221895, 0.0231465, -0.0413102, 0.0234261, -0.0456156, 0.0644566

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0712312, upper bound: 0.0688866
time: 3.15 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0712312, upper bound: 0.0725109
time: 3.45 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0408737, 0.0094401, -0.0470045, 0.0096570, -0.0505307, 0.0564446
1: 0.9317551, 1.0036863, 0.9314536, 1.0165949, -0.0848398, 0.0722327
2: -0.0168501, 0.0314081, -0.0170174, 0.0408539, -0.0577040, 0.0484255
3: -0.0259266, 0.0080975, -0.0315187, 0.0082412, -0.0341678, 0.0396163
4: -0.0244918, 0.0229276, -0.0320716, 0.0233747, -0.0478665, 0.0549992
5: -0.0060774, 0.0489587, -0.0068141, 0.0575600, -0.0636374, 0.0557728
6: -0.0102828, 0.0211351, -0.0106447, 0.0218023, -0.0320850, 0.0317798
7: -0.0112970, 0.0683942, -0.0114817, 0.0806263, -0.0919233, 0.0798759
8: -0.0067081, 0.0326099, -0.0072891, 0.0326544, -0.0393625, 0.0398991
9: -0.0333032, 0.0232098, -0.0451266, 0.0234916, -0.0567948, 0.0683365

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0626709, upper bound: 0.0711575
time: 1.35 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0718127, upper bound: 0.0712978
time: 1.51 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0451680, 0.0095883, -0.0177834, 0.0092964, -0.0544643, 0.0273717
1: 0.9315471, 1.0122955, 0.9717147, 0.9998229, -0.0429829, 0.0405807
2: -0.0169654, 0.0381836, -0.0168924, 0.0101695, -0.0271349, 0.0550759
3: -0.0298564, 0.0081966, -0.0124418, 0.0081361, -0.0379925, 0.0206383
4: -0.0297433, 0.0232359, -0.0071378, 0.0230498, -0.0527932, 0.0303737
5: -0.0063871, 0.0552426, -0.0050690, 0.0249830, -0.0313700, 0.0603116
6: -0.0105313, 0.0213638, -0.0102369, 0.0083786, -0.0189099, 0.0316006
7: -0.0114231, 0.0770954, -0.0111544, 0.0309208, -0.0423439, 0.0882497
8: -0.0071076, 0.0326406, -0.0066959, 0.0161204, -0.0232280, 0.0269204
9: -0.0418954, 0.0234041, -0.0128601, 0.0039129, -0.0458083, 0.0233486

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0653138, upper bound: 0.0662219
time: 1.97 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694993, upper bound: 0.0730472
time: 1.47 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0451680, 0.0095883, -0.0362368, 0.0094786, -0.0546466, 0.0458251
1: 0.9315471, 1.0122955, 0.9315278, 1.0027504, -0.0712033, 0.0807677
2: -0.0169654, 0.0381836, -0.0169722, 0.0249026, -0.0418680, 0.0551558
3: -0.0298564, 0.0081966, -0.0218689, 0.0082038, -0.0380602, 0.0300655
4: -0.0297433, 0.0232359, -0.0191761, 0.0232596, -0.0530029, 0.0424120
5: -0.0063871, 0.0552426, -0.0058844, 0.0428537, -0.0492408, 0.0611269
6: -0.0105313, 0.0213638, -0.0104641, 0.0210266, -0.0315580, 0.0318278
7: -0.0114231, 0.0770954, -0.0113176, 0.0593392, -0.0707623, 0.0884130
8: -0.0071076, 0.0326406, -0.0070363, 0.0326434, -0.0397510, 0.0396769
9: -0.0418954, 0.0234041, -0.0246697, 0.0234222, -0.0653176, 0.0480738

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0653138, upper bound: 0.0677698
time: 6.22 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694993, upper bound: 0.0740488
time: 1.53 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0356860, 0.0093896, -0.0384924, 0.0095304, -0.0452163, 0.0478820
1: 0.9316851, 1.0022829, 0.9315068, 1.0031613, -0.0714762, 0.0707760
2: -0.0168858, 0.0240440, -0.0169850, 0.0282126, -0.0450984, 0.0410290
3: -0.0213356, 0.0081293, -0.0238848, 0.0082144, -0.0295501, 0.0320141
4: -0.0183833, 0.0230274, -0.0218421, 0.0232922, -0.0416755, 0.0448695
5: -0.0055507, 0.0421817, -0.0060880, 0.0459481, -0.0514988, 0.0482697
6: -0.0102930, 0.0206684, -0.0105161, 0.0212012, -0.0314942, 0.0311845
7: -0.0112442, 0.0582746, -0.0113653, 0.0637961, -0.0750403, 0.0696399
8: -0.0067545, 0.0326203, -0.0071090, 0.0326465, -0.0394010, 0.0397293
9: -0.0237520, 0.0232753, -0.0289352, 0.0234418, -0.0471938, 0.0522105

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0674074, upper bound: 0.0724423
time: 2.05 seconds

## Relational analysis of NS_A2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0691934, upper bound: 0.0691934
time: 2.43 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0390226, 0.0094813, -0.0458554, 0.0096499, -0.0486725, 0.0553367
1: 0.9316180, 1.0029131, 0.9314473, 1.0139254, -0.0823074, 0.0714658
2: -0.0169243, 0.0290006, -0.0170205, 0.0391605, -0.0560848, 0.0460211
3: -0.0243345, 0.0081619, -0.0304882, 0.0082440, -0.0325785, 0.0386502
4: -0.0223720, 0.0231286, -0.0306760, 0.0233834, -0.0457553, 0.0538045
5: -0.0058861, 0.0467967, -0.0066989, 0.0560286, -0.0619147, 0.0534956
6: -0.0104052, 0.0209537, -0.0106436, 0.0217041, -0.0321094, 0.0315973
7: -0.0113265, 0.0649034, -0.0114745, 0.0783670, -0.0896935, 0.0763780
8: -0.0069218, 0.0326301, -0.0072909, 0.0326553, -0.0395771, 0.0399210
9: -0.0301028, 0.0233379, -0.0429756, 0.0234973, -0.0536001, 0.0663135

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0714398, upper bound: 0.0694992
time: 2.15 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714398, upper bound: 0.0731677
time: 1.76 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0451680, 0.0095883, -0.0384924, 0.0095304, -0.0546983, 0.0480807
1: 0.9315471, 1.0122955, 0.9315068, 1.0031613, -0.0716142, 0.0807887
2: -0.0169654, 0.0381836, -0.0169850, 0.0282126, -0.0451780, 0.0551686
3: -0.0298564, 0.0081966, -0.0238848, 0.0082144, -0.0380708, 0.0320814
4: -0.0297433, 0.0232359, -0.0218421, 0.0232922, -0.0530355, 0.0450779
5: -0.0063871, 0.0552426, -0.0060880, 0.0459481, -0.0523352, 0.0613306
6: -0.0105313, 0.0213638, -0.0105161, 0.0212012, -0.0317325, 0.0318799
7: -0.0114231, 0.0770954, -0.0113653, 0.0637961, -0.0752192, 0.0884607
8: -0.0071076, 0.0326406, -0.0071090, 0.0326465, -0.0397541, 0.0397496
9: -0.0418954, 0.0234041, -0.0289352, 0.0234418, -0.0653372, 0.0523393

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0625704, upper bound: 0.0693802
time: 1.45 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694255, upper bound: 0.0746253
time: 1.66 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0451680, 0.0095883, -0.0478514, 0.0097020, -0.0548700, 0.0574397
1: 0.9315471, 1.0122955, 0.9313772, 1.0185810, -0.0870339, 0.0809183
2: -0.0169654, 0.0381836, -0.0170595, 0.0421917, -0.0591572, 0.0552430
3: -0.0298564, 0.0081966, -0.0322955, 0.0082775, -0.0381338, 0.0404921
4: -0.0297433, 0.0232359, -0.0331239, 0.0234875, -0.0532308, 0.0563597
5: -0.0063871, 0.0552426, -0.0068993, 0.0587832, -0.0651703, 0.0621419
6: -0.0105313, 0.0213638, -0.0107291, 0.0218741, -0.0324054, 0.0320928
7: -0.0114231, 0.0770954, -0.0115191, 0.0823282, -0.0937513, 0.0886145
8: -0.0071076, 0.0326406, -0.0074276, 0.0326656, -0.0397732, 0.0400681
9: -0.0418954, 0.0234041, -0.0467909, 0.0235629, -0.0654583, 0.0701950

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0629775, upper bound: 0.0657342
time: 1.42 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0720400, upper bound: 0.0749625
time: 1.90 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.43 seconds
NS_A1_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691779, upper bound: 0.0733119
NS_A1_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691779, upper bound: 0.0733273
NS_A1_B1_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0673635, upper bound: 0.0673628
NS_A1_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0714235, upper bound: 0.0738269
NS_A1_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0692037, upper bound: 0.0747204
NS_A1_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0692037, upper bound: 0.0747204
NS_A1_B1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0720837
NS_A1_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0734531
NS_A1_B1_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698643, upper bound: 0.0694014
NS_A1_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0732190, upper bound: 0.0747076
NS_A1_B1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0722271
NS_A1_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0734754
NS_A1_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0743420
NS_A1_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698111, upper bound: 0.0752804
NS_A1_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691023, upper bound: 0.0732785
NS_A1_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691023, upper bound: 0.0732979
NS_A1_B2_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0673582, upper bound: 0.0673519
NS_A1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0713885, upper bound: 0.0737857
NS_A1_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691275, upper bound: 0.0746696
NS_A1_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691275, upper bound: 0.0750428
NS_A1_B2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0720353
NS_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0734218
NS_A1_B2_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0698229, upper bound: 0.0693732
NS_A1_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0731545, upper bound: 0.0746605
NS_A1_B2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0721869
NS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0734507
NS_A1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0743108
NS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0697009, upper bound: 0.0752484
NS_A2_A1_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0692803, upper bound: 0.0684439
NS_A2_A1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0729591, upper bound: 0.0737441
NS_A2_A1_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0712312, upper bound: 0.0688866
NS_A2_A1_B2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0712312, upper bound: 0.0725109
NS_A2_A1_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0626709, upper bound: 0.0711575
NS_A2_A1_B2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0718127, upper bound: 0.0712978
NS_A2_A2_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0653138, upper bound: 0.0662219
NS_A2_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0694993, upper bound: 0.0730472
NS_A2_A2_B1_A2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0653138, upper bound: 0.0677698
NS_A2_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0694993, upper bound: 0.0740488
NS_A2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0674074, upper bound: 0.0724423
NS_A2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0691934, upper bound: 0.0691934
NS_A2_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0714398, upper bound: 0.0694992
NS_A2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0714398, upper bound: 0.0731677
NS_A2_A2_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0625704, upper bound: 0.0693802
NS_A2_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0694255, upper bound: 0.0746253
NS_A2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0629775, upper bound: 0.0657342
NS_A2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.43
Output dim: 1, lower bound: -0.0720400, upper bound: 0.0749625

## BFS NS instance: NS_A1_B1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0296256, 0.0090009, -0.0460442, 0.0390107
1: 0.9322081, 1.0022494, 0.9714689, 1.0010142, -0.0392608, 0.0307805
2: -0.0168533, 0.0259523, -0.0165537, 0.0164484, -0.0333017, 0.0425060
3: -0.0225295, 0.0081010, -0.0168668, 0.0078435, -0.0303730, 0.0249678
4: -0.0198907, 0.0229389, -0.0123891, 0.0221379, -0.0420286, 0.0353281
5: -0.0054429, 0.0440414, -0.0049654, 0.0343344, -0.0397773, 0.0490068
6: -0.0102488, 0.0205214, -0.0096034, 0.0107600, -0.0210087, 0.0301248
7: -0.0112441, 0.0610160, -0.0109173, 0.0473493, -0.0585934, 0.0719333
8: -0.0066718, 0.0326113, -0.0056344, 0.0167083, -0.0233801, 0.0244499
9: -0.0264003, 0.0232185, -0.0155839, 0.0033146, -0.0297149, 0.0259570

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733119
time: 2.57 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0400096, 0.0092293, -0.0462727, 0.0493947
1: 0.9317459, 1.0022494, 0.9321564, 1.0027177, -0.0709718, 0.0700930
2: -0.0168533, 0.0259523, -0.0166300, 0.0298607, -0.0467140, 0.0425823
3: -0.0225295, 0.0081010, -0.0250833, 0.0079076, -0.0304371, 0.0331842
4: -0.0198907, 0.0229389, -0.0232532, 0.0223359, -0.0422266, 0.0461921
5: -0.0054429, 0.0440414, -0.0057314, 0.0476466, -0.0530895, 0.0497728
6: -0.0102488, 0.0205214, -0.0098581, 0.0207867, -0.0310355, 0.0303795
7: -0.0112441, 0.0610160, -0.0111251, 0.0666597, -0.0779039, 0.0721411
8: -0.0066718, 0.0326113, -0.0060033, 0.0325509, -0.0392227, 0.0386146
9: -0.0264003, 0.0232185, -0.0316371, 0.0228350, -0.0492353, 0.0548556

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0669082, upper bound: 0.0695509
time: 2.00 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725434, upper bound: 0.0730910
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0443733, 0.0095081, -0.0323475, 0.0090902, -0.0534635, 0.0418556
1: 0.9316862, 1.0104015, 0.9321362, 1.0013850, -0.0696988, 0.0782653
2: -0.0168890, 0.0368216, -0.0166364, 0.0187601, -0.0356490, 0.0534580
3: -0.0290979, 0.0081307, -0.0182519, 0.0079148, -0.0370127, 0.0263825
4: -0.0286432, 0.0230306, -0.0142305, 0.0223598, -0.0510029, 0.0372611
5: -0.0060837, 0.0540976, -0.0051064, 0.0373466, -0.0434303, 0.0592040
6: -0.0103790, 0.0210536, -0.0097700, 0.0202635, -0.0306425, 0.0308236
7: -0.0113568, 0.0755260, -0.0109916, 0.0515266, -0.0628833, 0.0865175
8: -0.0068571, 0.0326201, -0.0059074, 0.0325539, -0.0394110, 0.0385275
9: -0.0403774, 0.0232742, -0.0172842, 0.0228539, -0.0632313, 0.0405584

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714137, upper bound: 0.0738125
time: 1.35 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714137, upper bound: 0.0738193
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0296256, 0.0090009, -0.0553703, 0.0391865
1: 0.9316154, 1.0150608, 0.9714689, 1.0010142, -0.0415652, 0.0435919
2: -0.0169283, 0.0398458, -0.0165537, 0.0164484, -0.0333767, 0.0563994
3: -0.0309045, 0.0081644, -0.0168668, 0.0078435, -0.0387480, 0.0250313
4: -0.0310859, 0.0231355, -0.0123891, 0.0221379, -0.0532239, 0.0355247
5: -0.0062871, 0.0568534, -0.0049654, 0.0343344, -0.0406215, 0.0618188
6: -0.0104653, 0.0212261, -0.0096034, 0.0107600, -0.0212253, 0.0308295
7: -0.0114019, 0.0794904, -0.0109173, 0.0473493, -0.0587511, 0.0904077
8: -0.0069951, 0.0326305, -0.0056344, 0.0167083, -0.0237034, 0.0259583
9: -0.0441951, 0.0233403, -0.0155839, 0.0033146, -0.0475097, 0.0261306

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692273, upper bound: 0.0690562
time: 1.50 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725654, upper bound: 0.0744861
time: 1.56 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0400096, 0.0092293, -0.0555988, 0.0495704
1: 0.9316154, 1.0150608, 0.9321564, 1.0027177, -0.0711023, 0.0829044
2: -0.0169283, 0.0398458, -0.0166300, 0.0298607, -0.0467889, 0.0564758
3: -0.0309045, 0.0081644, -0.0250833, 0.0079076, -0.0388121, 0.0332477
4: -0.0310859, 0.0231355, -0.0232532, 0.0223359, -0.0534219, 0.0463887
5: -0.0062871, 0.0568534, -0.0057314, 0.0476466, -0.0539337, 0.0625847
6: -0.0104653, 0.0212261, -0.0098581, 0.0207867, -0.0312520, 0.0310842
7: -0.0114019, 0.0794904, -0.0111251, 0.0666597, -0.0780616, 0.0906155
8: -0.0069951, 0.0326305, -0.0060033, 0.0325509, -0.0395459, 0.0386338
9: -0.0441951, 0.0233403, -0.0316371, 0.0228350, -0.0670301, 0.0549774

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692273, upper bound: 0.0690562
time: 1.47 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725654, upper bound: 0.0748387
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0348758, 0.0091955, -0.0470609, 0.0443118
1: 0.9316702, 1.0024072, 0.9320502, 1.0017948, -0.0701246, 0.0703570
2: -0.0168951, 0.0272271, -0.0166854, 0.0225802, -0.0394753, 0.0439125
3: -0.0232820, 0.0081369, -0.0205412, 0.0079564, -0.0312384, 0.0286782
4: -0.0208966, 0.0230509, -0.0172242, 0.0224889, -0.0433855, 0.0402751
5: -0.0055243, 0.0452254, -0.0052229, 0.0409243, -0.0464486, 0.0504483
6: -0.0103369, 0.0205902, -0.0099049, 0.0203374, -0.0306743, 0.0304951
7: -0.0112871, 0.0626766, -0.0110855, 0.0566423, -0.0679294, 0.0737621
8: -0.0068144, 0.0326225, -0.0061111, 0.0325665, -0.0393809, 0.0387336
9: -0.0280082, 0.0232892, -0.0221898, 0.0229342, -0.0509424, 0.0454790

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0650106, upper bound: 0.0676012
time: 1.56 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0695896, upper bound: 0.0732190
time: 2.07 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0452187, 0.0095535, -0.0381902, 0.0092888, -0.0545075, 0.0477437
1: 0.9316103, 1.0123731, 0.9319821, 1.0024300, -0.0708197, 0.0803910
2: -0.0169307, 0.0381518, -0.0167246, 0.0275023, -0.0444330, 0.0548764
3: -0.0298724, 0.0081667, -0.0235190, 0.0079896, -0.0378620, 0.0316857
4: -0.0296891, 0.0231427, -0.0211814, 0.0225917, -0.0522808, 0.0443240
5: -0.0061699, 0.0553169, -0.0055594, 0.0455038, -0.0516736, 0.0608762
6: -0.0104633, 0.0211258, -0.0100191, 0.0206247, -0.0310880, 0.0311448
7: -0.0113945, 0.0772272, -0.0111693, 0.0632219, -0.0746164, 0.0883964
8: -0.0069952, 0.0326313, -0.0062813, 0.0325765, -0.0395718, 0.0389125
9: -0.0420399, 0.0233451, -0.0284906, 0.0229979, -0.0650378, 0.0518357

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696182, upper bound: 0.0731121
time: 1.68 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696182, upper bound: 0.0747076
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0443065, 0.0093966, -0.0472620, 0.0537425
1: 0.9316702, 1.0024072, 0.9319113, 1.0102394, -0.0785692, 0.0704958
2: -0.0168951, 0.0272271, -0.0167657, 0.0366005, -0.0534956, 0.0439928
3: -0.0232820, 0.0081369, -0.0290096, 0.0080242, -0.0313063, 0.0371466
4: -0.0208966, 0.0230509, -0.0285139, 0.0226989, -0.0435955, 0.0515648
5: -0.0055243, 0.0452254, -0.0060570, 0.0538829, -0.0594071, 0.0512823
6: -0.0103369, 0.0205902, -0.0101456, 0.0210356, -0.0313725, 0.0307358
7: -0.0112871, 0.0626766, -0.0112666, 0.0753484, -0.0866355, 0.0739432
8: -0.0068144, 0.0326225, -0.0064676, 0.0325870, -0.0394014, 0.0390900
9: -0.0280082, 0.0232892, -0.0401719, 0.0230640, -0.0510722, 0.0634611

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0624430, upper bound: 0.0700770
time: 1.62 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716678, upper bound: 0.0732425
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0328123, 0.0091396, -0.0563551, 0.0424176
1: 0.9315395, 1.0170586, 0.9320630, 1.0013683, -0.0698287, 0.0849956
2: -0.0169700, 0.0411812, -0.0166768, 0.0195860, -0.0365560, 0.0578580
3: -0.0316799, 0.0082004, -0.0186993, 0.0079495, -0.0396294, 0.0268997
4: -0.0321350, 0.0232476, -0.0148039, 0.0224680, -0.0546030, 0.0380516
5: -0.0063744, 0.0580774, -0.0050383, 0.0381370, -0.0445114, 0.0631157
6: -0.0105490, 0.0212991, -0.0098554, 0.0201788, -0.0307278, 0.0311545
7: -0.0114388, 0.0811907, -0.0110334, 0.0525543, -0.0639931, 0.0922241
8: -0.0071325, 0.0326417, -0.0060454, 0.0325646, -0.0396971, 0.0386871
9: -0.0458587, 0.0234112, -0.0183132, 0.0229222, -0.0687809, 0.0417244

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0678564, upper bound: 0.0677870
time: 1.59 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721390, upper bound: 0.0741120
time: 2.20 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0443065, 0.0093966, -0.0566120, 0.0539119
1: 0.9315395, 1.0170586, 0.9319113, 1.0102394, -0.0786998, 0.0851473
2: -0.0169700, 0.0411812, -0.0167657, 0.0366005, -0.0535705, 0.0579469
3: -0.0316799, 0.0082004, -0.0290096, 0.0080242, -0.0397041, 0.0372101
4: -0.0321350, 0.0232476, -0.0285139, 0.0226989, -0.0548339, 0.0517615
5: -0.0063744, 0.0580774, -0.0060570, 0.0538829, -0.0602572, 0.0641344
6: -0.0105490, 0.0212991, -0.0101456, 0.0210356, -0.0315846, 0.0314448
7: -0.0114388, 0.0811907, -0.0112666, 0.0753484, -0.0867872, 0.0924573
8: -0.0071325, 0.0326417, -0.0064676, 0.0325870, -0.0397194, 0.0391093
9: -0.0458587, 0.0234112, -0.0401719, 0.0230640, -0.0689227, 0.0635832

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 121

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0678564, upper bound: 0.0697018
time: 1.88 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721390, upper bound: 0.0750365
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0315084, 0.0091900, -0.0462334, 0.0408935
1: 0.9317459, 1.0022494, 0.9708926, 1.0015582, -0.0396804, 0.0313568
2: -0.0168533, 0.0259523, -0.0167705, 0.0179910, -0.0348443, 0.0427228
3: -0.0225295, 0.0081010, -0.0177193, 0.0080308, -0.0305603, 0.0258203
4: -0.0198907, 0.0229389, -0.0136512, 0.0227216, -0.0426123, 0.0365901
5: -0.0054429, 0.0440414, -0.0053181, 0.0360908, -0.0415337, 0.0493596
6: -0.0102488, 0.0205214, -0.0100089, 0.0113982, -0.0216469, 0.0305302
7: -0.0112441, 0.0610160, -0.0110691, 0.0500361, -0.0612803, 0.0720850
8: -0.0066718, 0.0326113, -0.0063139, 0.0168764, -0.0235483, 0.0251274
9: -0.0264003, 0.0232185, -0.0162404, 0.0036976, -0.0300979, 0.0266268

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0689798, upper bound: 0.0674281
time: 3.09 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0724802, upper bound: 0.0730392
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0370433, 0.0093851, -0.0408737, 0.0094401, -0.0464834, 0.0502588
1: 0.9317459, 1.0022494, 0.9317551, 1.0036863, -0.0719404, 0.0704943
2: -0.0168533, 0.0259523, -0.0168501, 0.0314081, -0.0482614, 0.0428024
3: -0.0225295, 0.0081010, -0.0259266, 0.0080975, -0.0306270, 0.0340275
4: -0.0198907, 0.0229389, -0.0244918, 0.0229276, -0.0428183, 0.0474307
5: -0.0054429, 0.0440414, -0.0060774, 0.0489587, -0.0544016, 0.0501188
6: -0.0102488, 0.0205214, -0.0102828, 0.0211351, -0.0313839, 0.0308041
7: -0.0112441, 0.0610160, -0.0112970, 0.0683942, -0.0796383, 0.0723130
8: -0.0066718, 0.0326113, -0.0067081, 0.0326099, -0.0392818, 0.0393194
9: -0.0264003, 0.0232185, -0.0333032, 0.0232098, -0.0496101, 0.0565217

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0668825, upper bound: 0.0695506
time: 1.70 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0724802, upper bound: 0.0730592
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0443733, 0.0095081, -0.0333081, 0.0093131, -0.0536864, 0.0428163
1: 0.9316862, 1.0104015, 0.9317247, 1.0019433, -0.0702572, 0.0786768
2: -0.0168890, 0.0368216, -0.0168623, 0.0204524, -0.0373413, 0.0536839
3: -0.0290979, 0.0081307, -0.0191909, 0.0081097, -0.0372076, 0.0273216
4: -0.0286432, 0.0230306, -0.0155869, 0.0229668, -0.0516100, 0.0386175
5: -0.0060837, 0.0540976, -0.0054681, 0.0387902, -0.0448738, 0.0595657
6: -0.0103790, 0.0210536, -0.0102104, 0.0206286, -0.0310076, 0.0312640
7: -0.0113568, 0.0755260, -0.0111743, 0.0534653, -0.0648220, 0.0867003
8: -0.0068571, 0.0326201, -0.0066361, 0.0326144, -0.0394716, 0.0392562
9: -0.0403774, 0.0232742, -0.0190954, 0.0232383, -0.0636157, 0.0423697

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0713795, upper bound: 0.0737705
time: 1.61 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0713795, upper bound: 0.0737796
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0315084, 0.0091900, -0.0555595, 0.0410693
1: 0.9316154, 1.0150608, 0.9708926, 1.0015582, -0.0419848, 0.0441682
2: -0.0169283, 0.0398458, -0.0167705, 0.0179910, -0.0349192, 0.0566162
3: -0.0309045, 0.0081644, -0.0177193, 0.0080308, -0.0389353, 0.0258837
4: -0.0310859, 0.0231355, -0.0136512, 0.0227216, -0.0538075, 0.0367867
5: -0.0062871, 0.0568534, -0.0053181, 0.0360908, -0.0423779, 0.0621715
6: -0.0104653, 0.0212261, -0.0100089, 0.0113982, -0.0218635, 0.0312350
7: -0.0114019, 0.0794904, -0.0110691, 0.0500361, -0.0614380, 0.0905595
8: -0.0069951, 0.0326305, -0.0063139, 0.0168764, -0.0238715, 0.0266357
9: -0.0441951, 0.0233403, -0.0162404, 0.0036976, -0.0478926, 0.0268004

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0691868, upper bound: 0.0690012
time: 2.56 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725017, upper bound: 0.0744351
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0463695, 0.0095609, -0.0408737, 0.0094401, -0.0558096, 0.0504346
1: 0.9316154, 1.0150608, 0.9317551, 1.0036863, -0.0720709, 0.0833057
2: -0.0169283, 0.0398458, -0.0168501, 0.0314081, -0.0483364, 0.0566959
3: -0.0309045, 0.0081644, -0.0259266, 0.0080975, -0.0390020, 0.0340910
4: -0.0310859, 0.0231355, -0.0244918, 0.0229276, -0.0540135, 0.0476273
5: -0.0062871, 0.0568534, -0.0060774, 0.0489587, -0.0552458, 0.0629308
6: -0.0104653, 0.0212261, -0.0102828, 0.0211351, -0.0316004, 0.0315089
7: -0.0114019, 0.0794904, -0.0112970, 0.0683942, -0.0797960, 0.0907874
8: -0.0069951, 0.0326305, -0.0067081, 0.0326099, -0.0396050, 0.0393386
9: -0.0441951, 0.0233403, -0.0333032, 0.0232098, -0.0674049, 0.0566436

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0691868, upper bound: 0.0694879
time: 2.47 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725017, upper bound: 0.0748019
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0356860, 0.0093896, -0.0472550, 0.0451219
1: 0.9316702, 1.0024072, 0.9316851, 1.0022829, -0.0706127, 0.0707220
2: -0.0168951, 0.0272271, -0.0168858, 0.0240440, -0.0409391, 0.0441129
3: -0.0232820, 0.0081369, -0.0213356, 0.0081293, -0.0314113, 0.0294726
4: -0.0208966, 0.0230509, -0.0183833, 0.0230274, -0.0439240, 0.0414342
5: -0.0055243, 0.0452254, -0.0055507, 0.0421817, -0.0477060, 0.0507761
6: -0.0103369, 0.0205902, -0.0102930, 0.0206684, -0.0310053, 0.0308832
7: -0.0112871, 0.0626766, -0.0112442, 0.0582746, -0.0695617, 0.0739208
8: -0.0068144, 0.0326225, -0.0067545, 0.0326203, -0.0394347, 0.0393770
9: -0.0280082, 0.0232892, -0.0237520, 0.0232753, -0.0512835, 0.0470411

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0649629, upper bound: 0.0675833
time: 3.02 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694808, upper bound: 0.0731820
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0452187, 0.0095535, -0.0390226, 0.0094813, -0.0547000, 0.0485761
1: 0.9316103, 1.0123731, 0.9316180, 1.0029131, -0.0713028, 0.0807551
2: -0.0169307, 0.0381518, -0.0169243, 0.0290006, -0.0459314, 0.0550761
3: -0.0298724, 0.0081667, -0.0243345, 0.0081619, -0.0380343, 0.0325011
4: -0.0296891, 0.0231427, -0.0223720, 0.0231286, -0.0528177, 0.0455146
5: -0.0061699, 0.0553169, -0.0058861, 0.0467967, -0.0529666, 0.0612030
6: -0.0104633, 0.0211258, -0.0104052, 0.0209537, -0.0314170, 0.0315310
7: -0.0113945, 0.0772272, -0.0113265, 0.0649034, -0.0762979, 0.0885536
8: -0.0069952, 0.0326313, -0.0069218, 0.0326301, -0.0396254, 0.0395531
9: -0.0420399, 0.0233451, -0.0301028, 0.0233379, -0.0653778, 0.0534479

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0695071, upper bound: 0.0730649
time: 1.66 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0695071, upper bound: 0.0746605
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0378654, 0.0094359, -0.0451680, 0.0095883, -0.0474537, 0.0546039
1: 0.9316702, 1.0024072, 0.9315471, 1.0122955, -0.0806253, 0.0708601
2: -0.0168951, 0.0272271, -0.0169654, 0.0381836, -0.0550787, 0.0441925
3: -0.0232820, 0.0081369, -0.0298564, 0.0081966, -0.0314786, 0.0379933
4: -0.0208966, 0.0230509, -0.0297433, 0.0232359, -0.0441324, 0.0527942
5: -0.0055243, 0.0452254, -0.0063871, 0.0552426, -0.0607668, 0.0516124
6: -0.0103369, 0.0205902, -0.0105313, 0.0213638, -0.0317006, 0.0311215
7: -0.0112871, 0.0626766, -0.0114231, 0.0770954, -0.0883825, 0.0740997
8: -0.0068144, 0.0326225, -0.0071076, 0.0326406, -0.0394550, 0.0397300
9: -0.0280082, 0.0232892, -0.0418954, 0.0234041, -0.0514123, 0.0651846

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 102

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0625638, upper bound: 0.0700660
time: 1.62 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716594, upper bound: 0.0732118
time: 2.85 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0337204, 0.0093416, -0.0565570, 0.0433258
1: 0.9315395, 1.0170586, 0.9316908, 1.0018823, -0.0703428, 0.0853678
2: -0.0169700, 0.0411812, -0.0168812, 0.0211895, -0.0381595, 0.0580624
3: -0.0316799, 0.0082004, -0.0195864, 0.0081258, -0.0398057, 0.0277868
4: -0.0321350, 0.0232476, -0.0160861, 0.0230172, -0.0551522, 0.0393337
5: -0.0063744, 0.0580774, -0.0053724, 0.0395139, -0.0458883, 0.0634499
6: -0.0105490, 0.0212991, -0.0102540, 0.0205156, -0.0310646, 0.0315531
7: -0.0114388, 0.0811907, -0.0111989, 0.0543946, -0.0658334, 0.0923897
8: -0.0071325, 0.0326417, -0.0067048, 0.0326194, -0.0397519, 0.0393465
9: -0.0458587, 0.0234112, -0.0200527, 0.0232699, -0.0691286, 0.0434639

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0678502, upper bound: 0.0677589
time: 3.39 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721275, upper bound: 0.0740778
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0472155, 0.0096054, -0.0451680, 0.0095883, -0.0568038, 0.0547734
1: 0.9315395, 1.0170586, 0.9315471, 1.0122955, -0.0807559, 0.0855115
2: -0.0169700, 0.0411812, -0.0169654, 0.0381836, -0.0551536, 0.0581466
3: -0.0316799, 0.0082004, -0.0298564, 0.0081966, -0.0398765, 0.0380568
4: -0.0321350, 0.0232476, -0.0297433, 0.0232359, -0.0553709, 0.0529910
5: -0.0063744, 0.0580774, -0.0063871, 0.0552426, -0.0616169, 0.0644645
6: -0.0105490, 0.0212991, -0.0105313, 0.0213638, -0.0319127, 0.0318305
7: -0.0114388, 0.0811907, -0.0114231, 0.0770954, -0.0885342, 0.0926138
8: -0.0071325, 0.0326417, -0.0071076, 0.0326406, -0.0397730, 0.0397493
9: -0.0458587, 0.0234112, -0.0418954, 0.0234041, -0.0692628, 0.0653066

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 121

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0678502, upper bound: 0.0696924
time: 1.93 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721275, upper bound: 0.0750052
time: 1.71 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0389092, 0.0093778, -0.0376706, 0.0094792, -0.0483885, 0.0470484
1: 0.9318260, 1.0029020, 0.9315825, 1.0030050, -0.0711790, 0.0713195
2: -0.0168104, 0.0284712, -0.0169432, 0.0269368, -0.0437472, 0.0454143
3: -0.0241535, 0.0080636, -0.0231312, 0.0081784, -0.0323319, 0.0311948
4: -0.0221156, 0.0228221, -0.0208336, 0.0231801, -0.0452956, 0.0436556
5: -0.0058927, 0.0462509, -0.0060098, 0.0447655, -0.0506582, 0.0522608
6: -0.0101894, 0.0209784, -0.0104277, 0.0211351, -0.0313245, 0.0314060
7: -0.0112429, 0.0644874, -0.0113220, 0.0621349, -0.0733778, 0.0758094
8: -0.0065615, 0.0325995, -0.0069660, 0.0326354, -0.0391969, 0.0395655
9: -0.0295535, 0.0231436, -0.0273296, 0.0233711, -0.0529246, 0.0504732

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0691727, upper bound: 0.0707333
time: 2.20 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729737, upper bound: 0.0737441
time: 1.58 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0432035, 0.0095338, -0.0177834, 0.0092964, -0.0524999, 0.0273172
1: 0.9316180, 1.0078115, 0.9717147, 0.9998229, -0.0424222, 0.0360968
2: -0.0169261, 0.0352015, -0.0168924, 0.0101695, -0.0270956, 0.0520939
3: -0.0280818, 0.0081628, -0.0124418, 0.0081361, -0.0362179, 0.0206046
4: -0.0273483, 0.0231308, -0.0071378, 0.0230498, -0.0503981, 0.0302686
5: -0.0061891, 0.0525207, -0.0050690, 0.0249830, -0.0311721, 0.0575897
6: -0.0104437, 0.0211965, -0.0102369, 0.0083786, -0.0188223, 0.0314334
7: -0.0113763, 0.0731981, -0.0111544, 0.0309208, -0.0422971, 0.0843525
8: -0.0069680, 0.0326301, -0.0066959, 0.0161204, -0.0230885, 0.0265996
9: -0.0381318, 0.0233380, -0.0128601, 0.0039129, -0.0420447, 0.0232334

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694928, upper bound: 0.0730408
time: 1.76 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694928, upper bound: 0.0730413
time: 1.72 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0432035, 0.0095338, -0.0362368, 0.0094786, -0.0526821, 0.0457706
1: 0.9316180, 1.0078115, 0.9315278, 1.0027504, -0.0711324, 0.0762838
2: -0.0169261, 0.0352015, -0.0169722, 0.0249026, -0.0418286, 0.0521738
3: -0.0280818, 0.0081628, -0.0218689, 0.0082038, -0.0362856, 0.0300318
4: -0.0273483, 0.0231308, -0.0191761, 0.0232596, -0.0506079, 0.0423069
5: -0.0061891, 0.0525207, -0.0058844, 0.0428537, -0.0490428, 0.0584051
6: -0.0104437, 0.0211965, -0.0104641, 0.0210266, -0.0314704, 0.0316605
7: -0.0113763, 0.0731981, -0.0113176, 0.0593392, -0.0707155, 0.0845158
8: -0.0069680, 0.0326301, -0.0070363, 0.0326434, -0.0396115, 0.0396665
9: -0.0381318, 0.0233380, -0.0246697, 0.0234222, -0.0615540, 0.0480076

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0713629, upper bound: 0.0714128
time: 1.67 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716976, upper bound: 0.0737361
time: 1.48 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0356860, 0.0093896, -0.0458554, 0.0096499, -0.0453358, 0.0552451
1: 0.9316851, 1.0022829, 0.9314473, 1.0139254, -0.0822403, 0.0708356
2: -0.0168858, 0.0240440, -0.0170205, 0.0391605, -0.0560463, 0.0410644
3: -0.0213356, 0.0081293, -0.0304882, 0.0082440, -0.0295796, 0.0386175
4: -0.0183833, 0.0230274, -0.0306760, 0.0233834, -0.0417667, 0.0537034
5: -0.0055507, 0.0421817, -0.0066989, 0.0560286, -0.0615793, 0.0488806
6: -0.0102930, 0.0206684, -0.0106436, 0.0217041, -0.0319971, 0.0313120
7: -0.0112442, 0.0582746, -0.0114745, 0.0783670, -0.0896112, 0.0697491
8: -0.0067545, 0.0326203, -0.0072909, 0.0326553, -0.0394098, 0.0399112
9: -0.0237520, 0.0232753, -0.0429756, 0.0234973, -0.0472493, 0.0662509

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692600, upper bound: 0.0722805
time: 1.80 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0710328, upper bound: 0.0729127
time: 1.71 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0432035, 0.0095338, -0.0384924, 0.0095304, -0.0527339, 0.0480261
1: 0.9316180, 1.0078115, 0.9315068, 1.0031613, -0.0715433, 0.0763047
2: -0.0169261, 0.0352015, -0.0169850, 0.0282126, -0.0451387, 0.0521866
3: -0.0280818, 0.0081628, -0.0238848, 0.0082144, -0.0362963, 0.0320476
4: -0.0273483, 0.0231308, -0.0218421, 0.0232922, -0.0506405, 0.0449728
5: -0.0061891, 0.0525207, -0.0060880, 0.0459481, -0.0521372, 0.0586087
6: -0.0104437, 0.0211965, -0.0105161, 0.0212012, -0.0316449, 0.0317126
7: -0.0113763, 0.0731981, -0.0113653, 0.0637961, -0.0751725, 0.0845634
8: -0.0069680, 0.0326301, -0.0071090, 0.0326465, -0.0396145, 0.0397391
9: -0.0381318, 0.0233380, -0.0289352, 0.0234418, -0.0615736, 0.0522732

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_A2_B1_A2_A2_A1

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0721024, upper bound: 0.0717107
time: 3.53 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_A2_A2_A2

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728885, upper bound: 0.0743529
time: 1.68 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0451680, 0.0095883, -0.0458554, 0.0096499, -0.0548178, 0.0554437
1: 0.9315471, 1.0122955, 0.9314473, 1.0139254, -0.0823783, 0.0808482
2: -0.0169654, 0.0381836, -0.0170205, 0.0391605, -0.0561259, 0.0552040
3: -0.0298564, 0.0081966, -0.0304882, 0.0082440, -0.0381004, 0.0386848
4: -0.0297433, 0.0232359, -0.0306760, 0.0233834, -0.0531267, 0.0539118
5: -0.0063871, 0.0552426, -0.0066989, 0.0560286, -0.0624156, 0.0619415
6: -0.0105313, 0.0213638, -0.0106436, 0.0217041, -0.0322355, 0.0320074
7: -0.0114231, 0.0770954, -0.0114745, 0.0783670, -0.0897901, 0.0885699
8: -0.0071076, 0.0326406, -0.0072909, 0.0326553, -0.0397628, 0.0399315
9: -0.0418954, 0.0234041, -0.0429756, 0.0234973, -0.0653927, 0.0663797

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 121

## Relational analysis of NS_A2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0710305, upper bound: 0.0727686
time: 1.69 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749151, upper bound: 0.0749625
time: 1.73 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 5.52 seconds
NS_A1_B1_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733119
NS_A1_B1_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
NS_A1_B1_B1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0669082, upper bound: 0.0695509
NS_A1_B1_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0725434, upper bound: 0.0730910
NS_A1_B1_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0714137, upper bound: 0.0738125
NS_A1_B1_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0714137, upper bound: 0.0738193
NS_A1_B1_B1_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0692273, upper bound: 0.0690562
NS_A1_B1_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0725654, upper bound: 0.0744861
NS_A1_B1_B1_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0692273, upper bound: 0.0690562
NS_A1_B1_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0725654, upper bound: 0.0748387
NS_A1_B1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0650106, upper bound: 0.0676012
NS_A1_B1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0695896, upper bound: 0.0732190
NS_A1_B1_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0696182, upper bound: 0.0731121
NS_A1_B1_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0696182, upper bound: 0.0747076
NS_A1_B1_B2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0624430, upper bound: 0.0700770
NS_A1_B1_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0716678, upper bound: 0.0732425
NS_A1_B1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0678564, upper bound: 0.0677870
NS_A1_B1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0721390, upper bound: 0.0741120
NS_A1_B1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0678564, upper bound: 0.0697018
NS_A1_B1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0721390, upper bound: 0.0750365
NS_A1_B2_B1_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0689798, upper bound: 0.0674281
NS_A1_B2_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0724802, upper bound: 0.0730392
NS_A1_B2_B1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0668825, upper bound: 0.0695506
NS_A1_B2_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0724802, upper bound: 0.0730592
NS_A1_B2_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0713795, upper bound: 0.0737705
NS_A1_B2_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0713795, upper bound: 0.0737796
NS_A1_B2_B1_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0691868, upper bound: 0.0690012
NS_A1_B2_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0725017, upper bound: 0.0744351
NS_A1_B2_B1_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0691868, upper bound: 0.0694879
NS_A1_B2_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0725017, upper bound: 0.0748019
NS_A1_B2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0649629, upper bound: 0.0675833
NS_A1_B2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0694808, upper bound: 0.0731820
NS_A1_B2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0695071, upper bound: 0.0730649
NS_A1_B2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0695071, upper bound: 0.0746605
NS_A1_B2_B2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0625638, upper bound: 0.0700660
NS_A1_B2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0716594, upper bound: 0.0732118
NS_A1_B2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0678502, upper bound: 0.0677589
NS_A1_B2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0721275, upper bound: 0.0740778
NS_A1_B2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0678502, upper bound: 0.0696924
NS_A1_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0721275, upper bound: 0.0750052
NS_A2_A1_B2_B1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0691727, upper bound: 0.0707333
NS_A2_A1_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0729737, upper bound: 0.0737441
NS_A2_A2_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0694928, upper bound: 0.0730408
NS_A2_A2_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0694928, upper bound: 0.0730413
NS_A2_A2_B1_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0713629, upper bound: 0.0714128
NS_A2_A2_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0716976, upper bound: 0.0737361
NS_A2_A2_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0692600, upper bound: 0.0722805
NS_A2_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0710328, upper bound: 0.0729127
NS_A2_A2_B2_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0721024, upper bound: 0.0717107
NS_A2_A2_B2_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0728885, upper bound: 0.0743529
NS_A2_A2_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0710305, upper bound: 0.0727686
NS_A2_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.52
Output dim: 1, lower bound: -0.0749151, upper bound: 0.0749625

## BFS NS instance: NS_A1_B1_B1_A2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0370058, 0.0093811, -0.0255790, 0.0089134, -0.0459192, 0.0349601
1: 0.9331356, 1.0022182, 0.9720066, 1.0002811, -0.0388710, 0.0302117
2: -0.0168497, 0.0258919, -0.0164534, 0.0137190, -0.0305687, 0.0423453
3: -0.0224938, 0.0080979, -0.0151901, 0.0077569, -0.0302507, 0.0232880
4: -0.0198379, 0.0229294, -0.0101393, 0.0218681, -0.0417060, 0.0330687
5: -0.0054214, 0.0439919, -0.0045832, 0.0307571, -0.0361786, 0.0485750
6: -0.0102415, 0.0204971, -0.0094159, 0.0097030, -0.0199444, 0.0299130
7: -0.0112408, 0.0609444, -0.0108471, 0.0416016, -0.0528423, 0.0717916
8: -0.0066599, 0.0326104, -0.0053203, 0.0164306, -0.0230905, 0.0241113
9: -0.0263354, 0.0232125, -0.0144135, 0.0031375, -0.0294729, 0.0247333

Time for backsubstitution: 1.61 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.56 + 595.89 = 600.45 seconds
