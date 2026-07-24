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
execution time: IAR + RelationalAnalysis = 0.79 + 2.77 = 3.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0767329, upper bound: 0.0767329

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764715, upper bound: 0.0764429
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764429, upper bound: 0.0764715
time: 1.59 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.75
Output dim: 1, lower bound: -0.0764715, upper bound: 0.0764429
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.75
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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 121

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764659, upper bound: 0.0764429
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764715, upper bound: 0.0764348
time: 1.26 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 121

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764348, upper bound: 0.0764715
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0764429, upper bound: 0.0764659
time: 1.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 1, lower bound: -0.0764659, upper bound: 0.0764429
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 1, lower bound: -0.0764715, upper bound: 0.0764348
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 1, lower bound: -0.0764348, upper bound: 0.0764715
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 1, lower bound: -0.0764429, upper bound: 0.0764659

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759362, upper bound: 0.0758675
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0759001, upper bound: 0.0759078
time: 1.97 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0757434, upper bound: 0.0757106
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0757434, upper bound: 0.0757106
time: 2.01 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0763255, upper bound: 0.0763624
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0763255, upper bound: 0.0763624
time: 1.55 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762623, upper bound: 0.0762177
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762147, upper bound: 0.0762893
time: 1.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0759362, upper bound: 0.0758675
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0759001, upper bound: 0.0759078
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0757434, upper bound: 0.0757106
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0757434, upper bound: 0.0757106
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0763255, upper bound: 0.0763624
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0763255, upper bound: 0.0763624
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0762623, upper bound: 0.0762177
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 1, lower bound: -0.0762147, upper bound: 0.0762893

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755738, upper bound: 0.0755038
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0755664, upper bound: 0.0755048
time: 1.26 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0758957, upper bound: 0.0758977
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0758948, upper bound: 0.0759034
time: 3.87 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0753821, upper bound: 0.0753508
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0753805, upper bound: 0.0753513
time: 1.63 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723206, upper bound: 0.0722942
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723206, upper bound: 0.0722942
time: 1.73 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726137, upper bound: 0.0726493
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726137, upper bound: 0.0726493
time: 1.08 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0760740, upper bound: 0.0761009
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0760741, upper bound: 0.0761009
time: 1.37 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762579, upper bound: 0.0762112
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762541, upper bound: 0.0762134
time: 1.75 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744322, upper bound: 0.0745023
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744322, upper bound: 0.0745023
time: 1.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0755738, upper bound: 0.0755038
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0755664, upper bound: 0.0755048
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0758957, upper bound: 0.0758977
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0758948, upper bound: 0.0759034
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0753821, upper bound: 0.0753508
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0753805, upper bound: 0.0753513
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0723206, upper bound: 0.0722942
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0723206, upper bound: 0.0722942
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0726137, upper bound: 0.0726493
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0726137, upper bound: 0.0726493
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0760740, upper bound: 0.0761009
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0760741, upper bound: 0.0761009
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0762579, upper bound: 0.0762112
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0762541, upper bound: 0.0762134
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0744322, upper bound: 0.0745023
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 1, lower bound: -0.0744322, upper bound: 0.0745023

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0629979, upper bound: 0.0629017
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0629979, upper bound: 0.0629017
time: 0.86 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737462, upper bound: 0.0736889
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737462, upper bound: 0.0736889
time: 1.64 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749827, upper bound: 0.0749962
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749827, upper bound: 0.0749962
time: 1.78 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692959, upper bound: 0.0693108
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692959, upper bound: 0.0693108
time: 1.11 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752820, upper bound: 0.0752495
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752819, upper bound: 0.0752501
time: 1.88 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0673592, upper bound: 0.0672267
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0673592, upper bound: 0.0672267
time: 1.05 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0756106, upper bound: 0.0756327
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0756046, upper bound: 0.0756426
time: 1.65 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0756108, upper bound: 0.0756327
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0756046, upper bound: 0.0756426
time: 1.40 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754259, upper bound: 0.0753844
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754259, upper bound: 0.0753844
time: 1.40 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0761509, upper bound: 0.0761120
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0761511, upper bound: 0.0761115
time: 1.26 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0641629, upper bound: 0.0643448
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0641629, upper bound: 0.0643448
time: 1.03 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0740983, upper bound: 0.0741639
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0740972, upper bound: 0.0741703
time: 3.10 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0629979, upper bound: 0.0629017
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0629979, upper bound: 0.0629017
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0737462, upper bound: 0.0736889
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0737462, upper bound: 0.0736889
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0749827, upper bound: 0.0749962
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0749827, upper bound: 0.0749962
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0692959, upper bound: 0.0693108
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0692959, upper bound: 0.0693108
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0752820, upper bound: 0.0752495
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0752819, upper bound: 0.0752501
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0673592, upper bound: 0.0672267
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0673592, upper bound: 0.0672267
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0756106, upper bound: 0.0756327
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0756046, upper bound: 0.0756426
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0756108, upper bound: 0.0756327
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0756046, upper bound: 0.0756426
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0754259, upper bound: 0.0753844
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0754259, upper bound: 0.0753844
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0761509, upper bound: 0.0761120
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0761511, upper bound: 0.0761115
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0641629, upper bound: 0.0643448
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0641629, upper bound: 0.0643448
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0740983, upper bound: 0.0741639
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 1, lower bound: -0.0740972, upper bound: 0.0741703

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0562209, upper bound: 0.0561765
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0562209, upper bound: 0.0561765
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737096, upper bound: 0.0736237
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736707, upper bound: 0.0736506
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0591511, upper bound: 0.0591612
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0591511, upper bound: 0.0591612
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730482, upper bound: 0.0730820
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730482, upper bound: 0.0730820
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752771, upper bound: 0.0752423
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752747, upper bound: 0.0752446
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0609947, upper bound: 0.0609615
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0609947, upper bound: 0.0609615
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754342, upper bound: 0.0753958
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0753956, upper bound: 0.0754596
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752465, upper bound: 0.0752860
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752408, upper bound: 0.0752918
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0611785, upper bound: 0.0611842
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0611785, upper bound: 0.0611842
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0750617, upper bound: 0.0750655
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0750215, upper bound: 0.0750980
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735342, upper bound: 0.0735042
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735342, upper bound: 0.0735042
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749513, upper bound: 0.0749029
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749438, upper bound: 0.0749150
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0696543, upper bound: 0.0696306
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0696543, upper bound: 0.0696306
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694197, upper bound: 0.0694069
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0694197, upper bound: 0.0694069
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0740016, upper bound: 0.0740684
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0740019, upper bound: 0.0740673
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735217, upper bound: 0.0735787
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735217, upper bound: 0.0735787
time: 1.40 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0562209, upper bound: 0.0561765
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0562209, upper bound: 0.0561765
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0737096, upper bound: 0.0736237
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0736707, upper bound: 0.0736506
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0591511, upper bound: 0.0591612
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0591511, upper bound: 0.0591612
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0730482, upper bound: 0.0730820
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0730482, upper bound: 0.0730820
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0752771, upper bound: 0.0752423
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0752747, upper bound: 0.0752446
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0609947, upper bound: 0.0609615
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0609947, upper bound: 0.0609615
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0754342, upper bound: 0.0753958
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0753956, upper bound: 0.0754596
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0752465, upper bound: 0.0752860
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0752408, upper bound: 0.0752918
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0611785, upper bound: 0.0611842
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0611785, upper bound: 0.0611842
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0750617, upper bound: 0.0750655
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0750215, upper bound: 0.0750980
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0735342, upper bound: 0.0735042
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0735342, upper bound: 0.0735042
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0749513, upper bound: 0.0749029
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0749438, upper bound: 0.0749150
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0696543, upper bound: 0.0696306
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0696543, upper bound: 0.0696306
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0694197, upper bound: 0.0694069
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0694197, upper bound: 0.0694069
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0740016, upper bound: 0.0740684
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0740019, upper bound: 0.0740673
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0735217, upper bound: 0.0735787
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 1, lower bound: -0.0735217, upper bound: 0.0735787

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727310, upper bound: 0.0726376
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727310, upper bound: 0.0726376
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0624404, upper bound: 0.0623112
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0624404, upper bound: 0.0623112
time: 2.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0720932, upper bound: 0.0721443
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0720932, upper bound: 0.0721443
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723964, upper bound: 0.0724564
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723964, upper bound: 0.0724564
time: 2.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741887, upper bound: 0.0741598
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741887, upper bound: 0.0741598
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752023, upper bound: 0.0751879
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752174, upper bound: 0.0751721
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754300, upper bound: 0.0753907
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754296, upper bound: 0.0753916
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0674039, upper bound: 0.0676074
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0674039, upper bound: 0.0676074
time: 2.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735521, upper bound: 0.0736353
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735521, upper bound: 0.0736353
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752365, upper bound: 0.0752870
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752353, upper bound: 0.0752876
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749665, upper bound: 0.0749773
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749727, upper bound: 0.0749768
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748466, upper bound: 0.0748585
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748091, upper bound: 0.0749275
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735342, upper bound: 0.0734893
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735101, upper bound: 0.0735039
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728686, upper bound: 0.0727993
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728686, upper bound: 0.0727993
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0705546, upper bound: 0.0704608
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0705546, upper bound: 0.0704608
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0592539, upper bound: 0.0592141
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0592539, upper bound: 0.0592141
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0701174, upper bound: 0.0703266
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0701174, upper bound: 0.0703266
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0739865, upper bound: 0.0740074
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0739601, upper bound: 0.0740486
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734366, upper bound: 0.0735230
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734635, upper bound: 0.0735112
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698827, upper bound: 0.0700478
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698827, upper bound: 0.0700478
time: 1.48 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0727310, upper bound: 0.0726376
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0727310, upper bound: 0.0726376
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0624404, upper bound: 0.0623112
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0624404, upper bound: 0.0623112
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0720932, upper bound: 0.0721443
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0720932, upper bound: 0.0721443
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0723964, upper bound: 0.0724564
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0723964, upper bound: 0.0724564
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0741887, upper bound: 0.0741598
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0741887, upper bound: 0.0741598
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0752023, upper bound: 0.0751879
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0752174, upper bound: 0.0751721
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0754300, upper bound: 0.0753907
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0754296, upper bound: 0.0753916
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0674039, upper bound: 0.0676074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0674039, upper bound: 0.0676074
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0735521, upper bound: 0.0736353
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0735521, upper bound: 0.0736353
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0752365, upper bound: 0.0752870
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0752353, upper bound: 0.0752876
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0749665, upper bound: 0.0749773
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0749727, upper bound: 0.0749768
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0748466, upper bound: 0.0748585
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0748091, upper bound: 0.0749275
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0735342, upper bound: 0.0734893
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0735101, upper bound: 0.0735039
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0728686, upper bound: 0.0727993
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0728686, upper bound: 0.0727993
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0705546, upper bound: 0.0704608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0705546, upper bound: 0.0704608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0592539, upper bound: 0.0592141
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0592539, upper bound: 0.0592141
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0701174, upper bound: 0.0703266
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0701174, upper bound: 0.0703266
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0739865, upper bound: 0.0740074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0739601, upper bound: 0.0740486
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0734366, upper bound: 0.0735230
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0734635, upper bound: 0.0735112
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0698827, upper bound: 0.0700478
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 1, lower bound: -0.0698827, upper bound: 0.0700478

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741802, upper bound: 0.0741356
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741578, upper bound: 0.0741526
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741802, upper bound: 0.0741356
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741578, upper bound: 0.0741526
time: 1.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748157, upper bound: 0.0747899
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0748084, upper bound: 0.0747981
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749638, upper bound: 0.0749235
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749645, upper bound: 0.0749209
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753756
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753756
time: 1.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753759
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753758
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726780, upper bound: 0.0727677
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726780, upper bound: 0.0727677
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0618043, upper bound: 0.0619180
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0618043, upper bound: 0.0619180
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0671370, upper bound: 0.0672051
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0671370, upper bound: 0.0672051
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752138, upper bound: 0.0752684
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752138, upper bound: 0.0752684
time: 3.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0664144, upper bound: 0.0664491
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0664144, upper bound: 0.0664491
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741502, upper bound: 0.0741569
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741502, upper bound: 0.0741569
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747771, upper bound: 0.0748008
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747855, upper bound: 0.0747780
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747926, upper bound: 0.0749141
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747926, upper bound: 0.0749141
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731462, upper bound: 0.0730940
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731399, upper bound: 0.0730973
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730358, upper bound: 0.0730226
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730295, upper bound: 0.0730332
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0701188, upper bound: 0.0702595
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0701188, upper bound: 0.0702595
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734825, upper bound: 0.0735631
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734772, upper bound: 0.0735721
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728886, upper bound: 0.0729666
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728397, upper bound: 0.0730117
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733525, upper bound: 0.0734215
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733538, upper bound: 0.0734210
time: 1.82 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.50 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0741802, upper bound: 0.0741356
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0741578, upper bound: 0.0741526
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0741802, upper bound: 0.0741356
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0741578, upper bound: 0.0741526
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0748157, upper bound: 0.0747899
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0748084, upper bound: 0.0747981
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0749638, upper bound: 0.0749235
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0749645, upper bound: 0.0749209
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753756
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753756
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753759
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0754155, upper bound: 0.0753758
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0726780, upper bound: 0.0727677
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0726780, upper bound: 0.0727677
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0618043, upper bound: 0.0619180
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0618043, upper bound: 0.0619180
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0671370, upper bound: 0.0672051
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0671370, upper bound: 0.0672051
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0752138, upper bound: 0.0752684
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0752138, upper bound: 0.0752684
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0664144, upper bound: 0.0664491
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0664144, upper bound: 0.0664491
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0741502, upper bound: 0.0741569
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0741502, upper bound: 0.0741569
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0747771, upper bound: 0.0748008
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0747855, upper bound: 0.0747780
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0747926, upper bound: 0.0749141
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0747926, upper bound: 0.0749141
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0731462, upper bound: 0.0730940
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0731399, upper bound: 0.0730973
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0730358, upper bound: 0.0730226
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0730295, upper bound: 0.0730332
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0701188, upper bound: 0.0702595
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0701188, upper bound: 0.0702595
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0734825, upper bound: 0.0735631
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0734772, upper bound: 0.0735721
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0728886, upper bound: 0.0729666
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0728397, upper bound: 0.0730117
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0733525, upper bound: 0.0734215
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.50
Output dim: 1, lower bound: -0.0733538, upper bound: 0.0734210

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738138, upper bound: 0.0737584
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738093, upper bound: 0.0737671
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0739074, upper bound: 0.0738949
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0739042, upper bound: 0.0739027
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738138, upper bound: 0.0737584
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738093, upper bound: 0.0737670
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736378, upper bound: 0.0736052
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736166, upper bound: 0.0736311
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0605739, upper bound: 0.0604969
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0605739, upper bound: 0.0604969
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0662515, upper bound: 0.0661474
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0662515, upper bound: 0.0661474
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0712270, upper bound: 0.0711268
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0712270, upper bound: 0.0711268
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747148, upper bound: 0.0746564
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747120, upper bound: 0.0746728
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0742112, upper bound: 0.0742003
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0742112, upper bound: 0.0742003
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0672723, upper bound: 0.0672716
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0672723, upper bound: 0.0672716
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0750338, upper bound: 0.0749898
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0750316, upper bound: 0.0749910
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0614646, upper bound: 0.0614073
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0614646, upper bound: 0.0614073
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0671075, upper bound: 0.0672951
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0671075, upper bound: 0.0672951
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0669902, upper bound: 0.0670741
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0669902, upper bound: 0.0670741
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737454, upper bound: 0.0737515
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737388, upper bound: 0.0737557
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733522, upper bound: 0.0733771
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733522, upper bound: 0.0733771
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730376, upper bound: 0.0731330
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730376, upper bound: 0.0731330
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0661766, upper bound: 0.0662513
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0661766, upper bound: 0.0662513
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0710266, upper bound: 0.0711473
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0710266, upper bound: 0.0711473
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0662577, upper bound: 0.0663764
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0662577, upper bound: 0.0663764
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0721802, upper bound: 0.0721149
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0721802, upper bound: 0.0721149
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726768, upper bound: 0.0726249
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726719, upper bound: 0.0726328
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728030, upper bound: 0.0727714
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727706, upper bound: 0.0727888
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727970, upper bound: 0.0727847
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727609, upper bound: 0.0727975
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726458, upper bound: 0.0727069
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726458, upper bound: 0.0727069
time: 1.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0655155, upper bound: 0.0655523
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0655155, upper bound: 0.0655523
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0725247, upper bound: 0.0726446
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0725432, upper bound: 0.0726429
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0724763, upper bound: 0.0727339
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0724879, upper bound: 0.0727297
time: 2.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728820, upper bound: 0.0729675
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728801, upper bound: 0.0729708
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0602789, upper bound: 0.0604196
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0602789, upper bound: 0.0604196
time: 1.02 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.82 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0738138, upper bound: 0.0737584
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0738093, upper bound: 0.0737671
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0739074, upper bound: 0.0738949
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0739042, upper bound: 0.0739027
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0738138, upper bound: 0.0737584
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0738093, upper bound: 0.0737670
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0736378, upper bound: 0.0736052
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0736166, upper bound: 0.0736311
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0605739, upper bound: 0.0604969
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0605739, upper bound: 0.0604969
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0662515, upper bound: 0.0661474
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0662515, upper bound: 0.0661474
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0712270, upper bound: 0.0711268
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0712270, upper bound: 0.0711268
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0747148, upper bound: 0.0746564
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0747120, upper bound: 0.0746728
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0742112, upper bound: 0.0742003
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0742112, upper bound: 0.0742003
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0672723, upper bound: 0.0672716
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0672723, upper bound: 0.0672716
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0750338, upper bound: 0.0749898
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0750316, upper bound: 0.0749910
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0614646, upper bound: 0.0614073
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0614646, upper bound: 0.0614073
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0671075, upper bound: 0.0672951
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0671075, upper bound: 0.0672951
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0669902, upper bound: 0.0670741
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0669902, upper bound: 0.0670741
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0737454, upper bound: 0.0737515
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0737388, upper bound: 0.0737557
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0733522, upper bound: 0.0733771
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0733522, upper bound: 0.0733771
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0730376, upper bound: 0.0731330
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0730376, upper bound: 0.0731330
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0661766, upper bound: 0.0662513
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0661766, upper bound: 0.0662513
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0710266, upper bound: 0.0711473
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0710266, upper bound: 0.0711473
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0662577, upper bound: 0.0663764
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0662577, upper bound: 0.0663764
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0721802, upper bound: 0.0721149
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0721802, upper bound: 0.0721149
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0726768, upper bound: 0.0726249
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0726719, upper bound: 0.0726328
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0728030, upper bound: 0.0727714
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0727706, upper bound: 0.0727888
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0727970, upper bound: 0.0727847
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0727609, upper bound: 0.0727975
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0726458, upper bound: 0.0727069
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0726458, upper bound: 0.0727069
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0655155, upper bound: 0.0655523
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0655155, upper bound: 0.0655523
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0725247, upper bound: 0.0726446
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0725432, upper bound: 0.0726429
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0724763, upper bound: 0.0727339
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0724879, upper bound: 0.0727297
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0728820, upper bound: 0.0729675
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0728801, upper bound: 0.0729708
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0602789, upper bound: 0.0604196
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 2.82
Output dim: 1, lower bound: -0.0602789, upper bound: 0.0604196

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737995, upper bound: 0.0737384
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737995, upper bound: 0.0737384
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726631, upper bound: 0.0726225
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0726631, upper bound: 0.0726225
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0641216, upper bound: 0.0639948
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0641216, upper bound: 0.0639948
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0724481, upper bound: 0.0724226
time: 2.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0724481, upper bound: 0.0724226
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736758, upper bound: 0.0735876
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736193, upper bound: 0.0736204
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737298, upper bound: 0.0737159
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737589, upper bound: 0.0736856
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0636298, upper bound: 0.0634732
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0636298, upper bound: 0.0634732
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736028, upper bound: 0.0736098
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736028, upper bound: 0.0736098
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0589970, upper bound: 0.0589499
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0589970, upper bound: 0.0589499
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741718, upper bound: 0.0741228
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0741538, upper bound: 0.0741293
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 239

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728613, upper bound: 0.0728438
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728613, upper bound: 0.0728438
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0702460, upper bound: 0.0702515
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0702460, upper bound: 0.0702515
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0743507, upper bound: 0.0743149
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0743507, upper bound: 0.0743149
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749257, upper bound: 0.0748879
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0749260, upper bound: 0.0748864
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736039, upper bound: 0.0735639
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735670, upper bound: 0.0736034
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729592, upper bound: 0.0729849
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729592, upper bound: 0.0729849
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
time: 1.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0635414, upper bound: 0.0636198
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0635414, upper bound: 0.0636198
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730300, upper bound: 0.0731325
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0730300, upper bound: 0.0731325
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722982, upper bound: 0.0723286
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722420, upper bound: 0.0724189
time: 2.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722980, upper bound: 0.0723292
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0722415, upper bound: 0.0724211
time: 4.16 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 6.67 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0737995, upper bound: 0.0737384
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0737995, upper bound: 0.0737384
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0726631, upper bound: 0.0726225
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0726631, upper bound: 0.0726225
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0641216, upper bound: 0.0639948
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0641216, upper bound: 0.0639948
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0724481, upper bound: 0.0724226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0724481, upper bound: 0.0724226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0736758, upper bound: 0.0735876
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0736193, upper bound: 0.0736204
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0737298, upper bound: 0.0737159
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0737589, upper bound: 0.0736856
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0636298, upper bound: 0.0634732
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0636298, upper bound: 0.0634732
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0736028, upper bound: 0.0736098
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0736028, upper bound: 0.0736098
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0589970, upper bound: 0.0589499
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0589970, upper bound: 0.0589499
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0741718, upper bound: 0.0741228
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0741538, upper bound: 0.0741293
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0728613, upper bound: 0.0728438
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0728613, upper bound: 0.0728438
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0702460, upper bound: 0.0702515
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0702460, upper bound: 0.0702515
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0743507, upper bound: 0.0743149
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0743507, upper bound: 0.0743149
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0749257, upper bound: 0.0748879
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0749260, upper bound: 0.0748864
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0736039, upper bound: 0.0735639
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0735670, upper bound: 0.0736034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0729592, upper bound: 0.0729849
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0729592, upper bound: 0.0729849
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0732880, upper bound: 0.0733136
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0635414, upper bound: 0.0636198
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0635414, upper bound: 0.0636198
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0730300, upper bound: 0.0731325
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0730300, upper bound: 0.0731325
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0722982, upper bound: 0.0723286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0722420, upper bound: 0.0724189
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0722980, upper bound: 0.0723292
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 6.67
Output dim: 1, lower bound: -0.0722415, upper bound: 0.0724211

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732555, upper bound: 0.0731839
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732305, upper bound: 0.0731933
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732555, upper bound: 0.0731839
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732305, upper bound: 0.0731933
time: 2.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736026, upper bound: 0.0735362
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0736264, upper bound: 0.0734849
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697436, upper bound: 0.0697042
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697436, upper bound: 0.0697042
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 244

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734435, upper bound: 0.0734343
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734459, upper bound: 0.0734283
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 244

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732402, upper bound: 0.0731309
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732180, upper bound: 0.0731593
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733544, upper bound: 0.0733563
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733511, upper bound: 0.0733614
time: 1.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732389, upper bound: 0.0732372
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732338, upper bound: 0.0732451
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.56 + 597.06 = 600.62 seconds
