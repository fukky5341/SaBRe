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
execution time: IAR + RelationalAnalysis = 1.76 + 2.87 = 4.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0767329, upper bound: 0.0767329

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764715, upper bound: 0.0764429
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764429, upper bound: 0.0764715
time: 1.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 1, lower bound: -0.0764715, upper bound: 0.0764429
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 1, lower bound: -0.0764429, upper bound: 0.0764715

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759707, upper bound: 0.0759295
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759550, upper bound: 0.0759393
time: 1.59 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759393, upper bound: 0.0759550
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759295, upper bound: 0.0759707
time: 1.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.80 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.80
Output dim: 1, lower bound: -0.0759707, upper bound: 0.0759295
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.80
Output dim: 1, lower bound: -0.0759550, upper bound: 0.0759393
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.80
Output dim: 1, lower bound: -0.0759393, upper bound: 0.0759550
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.80
Output dim: 1, lower bound: -0.0759295, upper bound: 0.0759707

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 121

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759651, upper bound: 0.0759295
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759707, upper bound: 0.0759207
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 121

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759468, upper bound: 0.0759393
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759550, upper bound: 0.0759315
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 121

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759315, upper bound: 0.0759550
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759393, upper bound: 0.0759468
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 121

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759207, upper bound: 0.0759707
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759295, upper bound: 0.0759651
time: 3.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 7.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759651, upper bound: 0.0759295
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759707, upper bound: 0.0759207
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759468, upper bound: 0.0759393
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759550, upper bound: 0.0759315
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759315, upper bound: 0.0759550
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759393, upper bound: 0.0759468
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759207, upper bound: 0.0759707
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.52
Output dim: 1, lower bound: -0.0759295, upper bound: 0.0759651

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755841, upper bound: 0.0755494
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755785, upper bound: 0.0755527
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755875, upper bound: 0.0755412
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755823, upper bound: 0.0755451
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755722, upper bound: 0.0755561
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755670, upper bound: 0.0755605
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755775, upper bound: 0.0755498
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755723, upper bound: 0.0755534
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755533, upper bound: 0.0755723
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755498, upper bound: 0.0755775
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755605, upper bound: 0.0755671
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755561, upper bound: 0.0755722
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755451, upper bound: 0.0755824
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755412, upper bound: 0.0755875
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755527, upper bound: 0.0755785
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755494, upper bound: 0.0755842
time: 1.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755841, upper bound: 0.0755494
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755785, upper bound: 0.0755527
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755875, upper bound: 0.0755412
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755823, upper bound: 0.0755451
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755722, upper bound: 0.0755561
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755670, upper bound: 0.0755605
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755775, upper bound: 0.0755498
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755723, upper bound: 0.0755534
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755533, upper bound: 0.0755723
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755498, upper bound: 0.0755775
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755605, upper bound: 0.0755671
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755561, upper bound: 0.0755722
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755451, upper bound: 0.0755824
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755412, upper bound: 0.0755875
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755527, upper bound: 0.0755785
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.09
Output dim: 1, lower bound: -0.0755494, upper bound: 0.0755842

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717675, upper bound: 0.0717250
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717675, upper bound: 0.0717250
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717642, upper bound: 0.0717265
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717642, upper bound: 0.0717265
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717711, upper bound: 0.0717233
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717711, upper bound: 0.0717233
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717681, upper bound: 0.0717248
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717681, upper bound: 0.0717248
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717512, upper bound: 0.0717332
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717512, upper bound: 0.0717332
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717496, upper bound: 0.0717365
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717496, upper bound: 0.0717365
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717530, upper bound: 0.0717310
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717530, upper bound: 0.0717310
time: 2.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717521, upper bound: 0.0717340
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717521, upper bound: 0.0717340
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717340, upper bound: 0.0717521
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717340, upper bound: 0.0717521
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717309, upper bound: 0.0717530
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717309, upper bound: 0.0717530
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717364, upper bound: 0.0717496
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717364, upper bound: 0.0717496
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717332, upper bound: 0.0717512
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717332, upper bound: 0.0717512
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717248, upper bound: 0.0717681
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717248, upper bound: 0.0717681
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717233, upper bound: 0.0717711
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717233, upper bound: 0.0717711
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717265, upper bound: 0.0717642
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717265, upper bound: 0.0717642
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835
1: 0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128
2: -0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145
3: -0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669
4: -0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407
5: -0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920
6: -0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919
7: -0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113
8: -0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782
9: -0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717250, upper bound: 0.0717675
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717250, upper bound: 0.0717675
time: 2.55 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717675, upper bound: 0.0717250
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717675, upper bound: 0.0717250
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717642, upper bound: 0.0717265
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717642, upper bound: 0.0717265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717711, upper bound: 0.0717233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717711, upper bound: 0.0717233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717681, upper bound: 0.0717248
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717681, upper bound: 0.0717248
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717512, upper bound: 0.0717332
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717512, upper bound: 0.0717332
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717496, upper bound: 0.0717365
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717496, upper bound: 0.0717365
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717530, upper bound: 0.0717310
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717530, upper bound: 0.0717310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717521, upper bound: 0.0717340
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717521, upper bound: 0.0717340
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717340, upper bound: 0.0717521
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717340, upper bound: 0.0717521
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717309, upper bound: 0.0717530
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717309, upper bound: 0.0717530
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717364, upper bound: 0.0717496
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717364, upper bound: 0.0717496
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717332, upper bound: 0.0717512
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717332, upper bound: 0.0717512
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717248, upper bound: 0.0717681
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717248, upper bound: 0.0717681
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717233, upper bound: 0.0717711
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717233, upper bound: 0.0717711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717265, upper bound: 0.0717642
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717265, upper bound: 0.0717642
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717250, upper bound: 0.0717675
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 6.72
Output dim: 1, lower bound: -0.0717250, upper bound: 0.0717675

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.63 + 154.96 = 159.58 seconds
