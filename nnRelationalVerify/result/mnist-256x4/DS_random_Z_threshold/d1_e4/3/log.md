## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0016185600000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013854)
1: (-0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035157)
2: (0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021812)
3: (0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040728, 0.0040728)
4: (-0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035761)
5: (0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013545)
6: (0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051689, 0.0051689)
7: (0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036170, 0.0036170)
8: (-0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038780, 0.0038780)
9: (-0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025616)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.91 + 2.67 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0020232, upper bound: 0.0020232

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019767, upper bound: 0.0019449
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019449, upper bound: 0.0019767
time: 1.85 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.21
Output dim: 7, lower bound: -0.0019767, upper bound: 0.0019449
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.21
Output dim: 7, lower bound: -0.0019449, upper bound: 0.0019767

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013098, 0.0013206
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033239, 0.0033511
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020622, 0.0020791
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038821, 0.0038506
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033809, 0.0034087
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012806, 0.0012911
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0049269, 0.0048869
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034476, 0.0034196
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036964, 0.0036663
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024218, 0.0024417

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019359, upper bound: 0.0018418
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018780, upper bound: 0.0019049
time: 1.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013206, 0.0013098
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0033511, 0.0033239
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020791, 0.0020622
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0038506, 0.0038821
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0034087, 0.0033809
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012911, 0.0012806
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0048869, 0.0049269
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0034196, 0.0034476
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0036663, 0.0036964
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0024417, 0.0024218

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019049, upper bound: 0.0018780
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018418, upper bound: 0.0019360
time: 1.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 7, lower bound: -0.0019359, upper bound: 0.0018418
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 7, lower bound: -0.0018780, upper bound: 0.0019049
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 7, lower bound: -0.0019049, upper bound: 0.0018780
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 7, lower bound: -0.0018418, upper bound: 0.0019360

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012428, 0.0012683
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031537, 0.0032185
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019565, 0.0019968
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037285, 0.0036534
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032078, 0.0032737
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012150, 0.0012400
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047319, 0.0046366
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033111, 0.0032445
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035501, 0.0034786
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022978, 0.0023450

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018806, upper bound: 0.0017828
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018799, upper bound: 0.0017828
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012585, 0.0012535
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031937, 0.0031809
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019814, 0.0019734
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036849, 0.0036998
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032485, 0.0032355
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012305, 0.0012255
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046766, 0.0046955
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032725, 0.0032857
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035086, 0.0035228
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023270, 0.0023176

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018467, upper bound: 0.0018841
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018567, upper bound: 0.0018747
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012535, 0.0012585
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031809, 0.0031937
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019734, 0.0019814
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036998, 0.0036849
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032355, 0.0032485
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012255, 0.0012305
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046955, 0.0046766
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032857, 0.0032725
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035228, 0.0035086
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023176, 0.0023270

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018923, upper bound: 0.0018625
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018904, upper bound: 0.0018621
time: 1.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012683, 0.0012428
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032185, 0.0031537
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019968, 0.0019565
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036534, 0.0037285
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032737, 0.0032078
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012400, 0.0012150
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046366, 0.0047319
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032445, 0.0033111
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034786, 0.0035501
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023450, 0.0022978

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018214, upper bound: 0.0018969
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018175, upper bound: 0.0019192
time: 1.19 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.69 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018806, upper bound: 0.0017828
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018799, upper bound: 0.0017828
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018467, upper bound: 0.0018841
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018567, upper bound: 0.0018747
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018923, upper bound: 0.0018625
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018904, upper bound: 0.0018621
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018214, upper bound: 0.0018969
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.69
Output dim: 7, lower bound: -0.0018175, upper bound: 0.0019192

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012335, 0.0012614
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031302, 0.0032010
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019420, 0.0019859
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037082, 0.0036262
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031840, 0.0032559
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012060, 0.0012333
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047062, 0.0046021
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032931, 0.0032204
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035308, 0.0034527
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022807, 0.0023323

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018799, upper bound: 0.0017820
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018782, upper bound: 0.0017819
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012428, 0.0012591
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031537, 0.0031950
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019565, 0.0019822
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037013, 0.0036534
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032078, 0.0032499
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012150, 0.0012310
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046974, 0.0046366
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032870, 0.0032445
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035242, 0.0034786
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022978, 0.0023280

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018357, upper bound: 0.0017432
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018386, upper bound: 0.0017374
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012252, 0.0012169
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031090, 0.0030881
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019288, 0.0019159
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035774, 0.0036017
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031624, 0.0031411
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011978, 0.0011898
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045402, 0.0045710
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031770, 0.0031985
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034063, 0.0034293
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022653, 0.0022500

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017849, upper bound: 0.0018248
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017843, upper bound: 0.0018248
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012220, 0.0012204
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031009, 0.0030969
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019238, 0.0019213
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035876, 0.0035923
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031542, 0.0031500
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011947, 0.0011932
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045531, 0.0045591
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031860, 0.0031902
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034159, 0.0034204
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022594, 0.0022564

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017775, upper bound: 0.0017844
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017775, upper bound: 0.0017844
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012509, 0.0012565
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031744, 0.0031885
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019694, 0.0019782
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036938, 0.0036774
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032289, 0.0032433
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012230, 0.0012285
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046879, 0.0046671
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032803, 0.0032658
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035171, 0.0035015
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023129, 0.0023232

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018907, upper bound: 0.0018582
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018905, upper bound: 0.0018611
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012513, 0.0012560
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031753, 0.0031873
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019700, 0.0019774
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036923, 0.0036784
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032298, 0.0032420
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012234, 0.0012280
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046860, 0.0046684
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032790, 0.0032667
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035156, 0.0035024
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023135, 0.0023223

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018056, upper bound: 0.0017809
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018056, upper bound: 0.0017810
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012662, 0.0012522
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032131, 0.0031777
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019934, 0.0019714
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036812, 0.0037222
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032683, 0.0032322
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012379, 0.0012243
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046719, 0.0047240
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032692, 0.0033056
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035051, 0.0035441
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023411, 0.0023153

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016730, upper bound: 0.0017507
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016730, upper bound: 0.0017507
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012792, 0.0012406
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032461, 0.0031483
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020139, 0.0019532
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036471, 0.0037604
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033018, 0.0032023
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012506, 0.0012130
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046287, 0.0047725
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032389, 0.0033395
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034726, 0.0035805
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023651, 0.0022939

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017613, upper bound: 0.0018650
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017612, upper bound: 0.0018665
time: 1.67 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018799, upper bound: 0.0017820
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018782, upper bound: 0.0017819
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018357, upper bound: 0.0017432
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018386, upper bound: 0.0017374
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0017849, upper bound: 0.0018248
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0017843, upper bound: 0.0018248
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0017775, upper bound: 0.0017844
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0017775, upper bound: 0.0017844
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018907, upper bound: 0.0018582
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018905, upper bound: 0.0018611
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018056, upper bound: 0.0017809
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0018056, upper bound: 0.0017810
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0016730, upper bound: 0.0017507
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0016730, upper bound: 0.0017507
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0017613, upper bound: 0.0018650
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 7, lower bound: -0.0017612, upper bound: 0.0018665

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012348, 0.0012629
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031335, 0.0032049
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019440, 0.0019883
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037127, 0.0036300
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031873, 0.0032599
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012073, 0.0012348
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047119, 0.0046069
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032972, 0.0032237
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035351, 0.0034563
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022831, 0.0023351

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018465, upper bound: 0.0017500
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018433, upper bound: 0.0017482
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012350, 0.0012627
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031340, 0.0032042
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019443, 0.0019879
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037120, 0.0036306
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031878, 0.0032592
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012074, 0.0012345
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047110, 0.0046076
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032965, 0.0032242
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035344, 0.0034569
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022835, 0.0023346

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016758, upper bound: 0.0015922
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016758, upper bound: 0.0015922
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011842, 0.0012018
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030050, 0.0030497
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018643, 0.0018921
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035329, 0.0034811
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030565, 0.0031021
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011577, 0.0011750
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044838, 0.0044180
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031375, 0.0030915
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033639, 0.0033145
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021894, 0.0022221

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016829, upper bound: 0.0015974
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016829, upper bound: 0.0015974
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011855, 0.0012010
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030083, 0.0030476
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018664, 0.0018908
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035306, 0.0034850
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030600, 0.0031000
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011590, 0.0011742
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044807, 0.0044229
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031354, 0.0030949
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033616, 0.0033183
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021919, 0.0022206

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018378, upper bound: 0.0017368
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018368, upper bound: 0.0017364
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012159, 0.0012103
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030856, 0.0030713
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019143, 0.0019054
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035579, 0.0035745
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031386, 0.0031240
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011888, 0.0011833
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045154, 0.0045365
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031597, 0.0031744
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033877, 0.0034035
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022482, 0.0022378

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017543, upper bound: 0.0017925
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017543, upper bound: 0.0017933
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012252, 0.0012077
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031090, 0.0030647
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019288, 0.0019013
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035503, 0.0036017
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031624, 0.0031173
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011978, 0.0011808
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045058, 0.0045710
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031529, 0.0031985
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033804, 0.0034293
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022653, 0.0022330

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017644, upper bound: 0.0018066
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017647, upper bound: 0.0018080
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012099, 0.0012111
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030703, 0.0030733
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019048, 0.0019067
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035602, 0.0035568
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031230, 0.0031260
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011829, 0.0011840
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045184, 0.0045140
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031617, 0.0031587
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033899, 0.0033866
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022370, 0.0022392

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017510, upper bound: 0.0017575
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017522, upper bound: 0.0017577
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012127, 0.0012204
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030773, 0.0030969
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019092, 0.0019213
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035876, 0.0035649
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031302, 0.0031500
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011856, 0.0011932
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045531, 0.0045244
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031860, 0.0031659
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034159, 0.0033944
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022422, 0.0022564

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017773, upper bound: 0.0017841
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017759, upper bound: 0.0017839
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012523, 0.0012581
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031779, 0.0031925
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019716, 0.0019806
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036984, 0.0036814
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032324, 0.0032473
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012244, 0.0012300
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046937, 0.0046722
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032844, 0.0032694
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035214, 0.0035053
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023154, 0.0023261

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018713, upper bound: 0.0018293
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018585, upper bound: 0.0018392
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012525, 0.0012579
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031785, 0.0031920
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019720, 0.0019803
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036978, 0.0036822
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032331, 0.0032468
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012246, 0.0012298
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046929, 0.0046731
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032839, 0.0032700
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035208, 0.0035060
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023159, 0.0023257

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017355, upper bound: 0.0017015
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017355, upper bound: 0.0017015
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012400, 0.0012465
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031466, 0.0031631
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019522, 0.0019624
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036643, 0.0036452
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032006, 0.0032174
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012123, 0.0012187
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046505, 0.0046262
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032542, 0.0032372
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034890, 0.0034708
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022927, 0.0023047

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018044, upper bound: 0.0017781
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018047, upper bound: 0.0017804
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012418, 0.0012560
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031511, 0.0031873
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019550, 0.0019774
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036923, 0.0036504
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032052, 0.0032420
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012141, 0.0012280
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046860, 0.0046329
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032790, 0.0032419
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035156, 0.0034758
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022960, 0.0023223

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016306, upper bound: 0.0015986
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016306, upper bound: 0.0015986
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012584, 0.0012445
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031933, 0.0031581
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019811, 0.0019593
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036585, 0.0036993
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032481, 0.0032123
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012303, 0.0012167
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046432, 0.0046948
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032491, 0.0032852
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034835, 0.0035223
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023267, 0.0023010

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016308, upper bound: 0.0017140
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016354, upper bound: 0.0017066
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012585, 0.0012522
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031935, 0.0031777
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019813, 0.0019714
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036812, 0.0036996
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032484, 0.0032322
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012304, 0.0012243
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046719, 0.0046952
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032692, 0.0032855
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035051, 0.0035226
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023269, 0.0023153

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016308, upper bound: 0.0017140
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016354, upper bound: 0.0017066
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012708, 0.0012349
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032249, 0.0031338
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020007, 0.0019442
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036304, 0.0037359
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032803, 0.0031876
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012425, 0.0012074
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046074, 0.0047413
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032240, 0.0033178
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034567, 0.0035572
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023497, 0.0022833

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017263, upper bound: 0.0018276
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017285, upper bound: 0.0018305
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012792, 0.0012323
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032461, 0.0031271
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0020139, 0.0019401
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036226, 0.0037604
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0033018, 0.0031808
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012506, 0.0012048
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045975, 0.0047725
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032171, 0.0033395
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034493, 0.0035805
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023651, 0.0022784

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017088, upper bound: 0.0018151
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017089, upper bound: 0.0018146
time: 1.43 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018465, upper bound: 0.0017500
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018433, upper bound: 0.0017482
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016758, upper bound: 0.0015922
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016758, upper bound: 0.0015922
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016829, upper bound: 0.0015974
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016829, upper bound: 0.0015974
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018378, upper bound: 0.0017368
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018368, upper bound: 0.0017364
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017543, upper bound: 0.0017925
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017543, upper bound: 0.0017933
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017644, upper bound: 0.0018066
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017647, upper bound: 0.0018080
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017510, upper bound: 0.0017575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017522, upper bound: 0.0017577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017773, upper bound: 0.0017841
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017759, upper bound: 0.0017839
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018713, upper bound: 0.0018293
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018585, upper bound: 0.0018392
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017355, upper bound: 0.0017015
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017355, upper bound: 0.0017015
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018044, upper bound: 0.0017781
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0018047, upper bound: 0.0017804
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016306, upper bound: 0.0015986
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016306, upper bound: 0.0015986
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016308, upper bound: 0.0017140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016354, upper bound: 0.0017066
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016308, upper bound: 0.0017140
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0016354, upper bound: 0.0017066
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017263, upper bound: 0.0018276
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017285, upper bound: 0.0018305
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017088, upper bound: 0.0018151
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 7, lower bound: -0.0017089, upper bound: 0.0018146

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010939, 0.0011349
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027760, 0.0028798
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017223, 0.0017867
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033362, 0.0032159
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028237, 0.0029293
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010695, 0.0011095
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042340, 0.0040814
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029628, 0.0028560
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031766, 0.0030620
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020226, 0.0020983

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017127, upper bound: 0.0015917
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016851, upper bound: 0.0016170
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011067, 0.0011193
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028084, 0.0028403
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017424, 0.0017621
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032903, 0.0032534
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028567, 0.0028890
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010820, 0.0010943
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041759, 0.0041290
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029221, 0.0028893
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031329, 0.0030978
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020463, 0.0020695

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018107, upper bound: 0.0017241
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018199, upper bound: 0.0017168
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012240, 0.0012533
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031060, 0.0031804
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019270, 0.0019732
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036844, 0.0035982
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031594, 0.0032350
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011967, 0.0012253
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046760, 0.0045666
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032720, 0.0031955
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035081, 0.0034260
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022631, 0.0023173

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016512, upper bound: 0.0015677
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016511, upper bound: 0.0015673
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012256, 0.0012627
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031102, 0.0032042
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019296, 0.0019879
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037120, 0.0036030
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031636, 0.0032592
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011983, 0.0012345
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047110, 0.0045727
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032965, 0.0031997
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035344, 0.0034306
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022661, 0.0023346

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014902, upper bound: 0.0014396
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014902, upper bound: 0.0014396
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011669, 0.0011877
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029613, 0.0030141
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018372, 0.0018699
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034917, 0.0034305
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030121, 0.0030658
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011409, 0.0011613
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044314, 0.0043538
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031009, 0.0030465
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033246, 0.0032664
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021576, 0.0021961

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016709, upper bound: 0.0015769
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016480, upper bound: 0.0015841
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011701, 0.0012018
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029693, 0.0030497
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018422, 0.0018921
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035329, 0.0034398
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030203, 0.0031021
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011440, 0.0011750
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044838, 0.0043656
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031375, 0.0030548
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033639, 0.0032752
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021635, 0.0022221

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0015035
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0015035
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011865, 0.0012022
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030109, 0.0030509
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018680, 0.0018928
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035343, 0.0034880
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030626, 0.0031033
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011600, 0.0011754
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044855, 0.0044268
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031387, 0.0030976
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033652, 0.0033212
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021938, 0.0022229

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018058, upper bound: 0.0017060
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018047, upper bound: 0.0017055
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011867, 0.0012020
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030114, 0.0030502
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018683, 0.0018924
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035335, 0.0034886
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030631, 0.0031026
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011602, 0.0011752
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044845, 0.0044275
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031380, 0.0030981
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033645, 0.0033217
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021942, 0.0022224

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017839, upper bound: 0.0016824
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017842, upper bound: 0.0016821
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010787, 0.0010877
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027373, 0.0027601
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016982, 0.0017124
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031974, 0.0031710
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027843, 0.0028075
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010546, 0.0010634
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040580, 0.0040245
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028396, 0.0028161
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030445, 0.0030193
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019944, 0.0020110

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016107, upper bound: 0.0016226
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015916, upper bound: 0.0016454
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010933, 0.0010749
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027744, 0.0027278
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017213, 0.0016923
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031600, 0.0032140
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028221, 0.0027746
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010689, 0.0010510
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040105, 0.0040790
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028064, 0.0028543
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030089, 0.0030603
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020215, 0.0019875

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016955, upper bound: 0.0017374
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016955, upper bound: 0.0017362
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012224, 0.0012051
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031019, 0.0030582
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019245, 0.0018973
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035428, 0.0035934
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031552, 0.0031107
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011951, 0.0011782
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044962, 0.0045605
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031462, 0.0031913
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033733, 0.0034215
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022601, 0.0022282

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017332, upper bound: 0.0017746
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017332, upper bound: 0.0017750
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012228, 0.0012048
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031030, 0.0030573
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019251, 0.0018968
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035418, 0.0035946
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031562, 0.0031098
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011955, 0.0011779
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044950, 0.0045621
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031454, 0.0031923
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033723, 0.0034227
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022609, 0.0022276

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017228, upper bound: 0.0017703
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017245, upper bound: 0.0017647
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010769, 0.0010922
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027328, 0.0027716
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016954, 0.0017195
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032108, 0.0031658
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027797, 0.0028192
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010529, 0.0010678
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040749, 0.0040178
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028514, 0.0028115
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030572, 0.0030143
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019911, 0.0020194

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016973, upper bound: 0.0017051
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016998, upper bound: 0.0017023
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010910, 0.0010811
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027686, 0.0027434
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017177, 0.0017020
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031782, 0.0032073
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028162, 0.0027906
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010667, 0.0010570
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040335, 0.0040705
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028224, 0.0028484
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030261, 0.0030539
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020173, 0.0019989

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016232, upper bound: 0.0016030
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015974, upper bound: 0.0016319
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012138, 0.0012217
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030801, 0.0031002
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019109, 0.0019234
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035914, 0.0035681
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031329, 0.0031534
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011867, 0.0011944
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045579, 0.0045284
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031894, 0.0031687
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034196, 0.0033974
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022442, 0.0022588

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017601, upper bound: 0.0017700
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017606, upper bound: 0.0017716
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012140, 0.0012215
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030806, 0.0030996
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019112, 0.0019230
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035908, 0.0035687
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031335, 0.0031529
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011869, 0.0011942
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045572, 0.0045292
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031889, 0.0031693
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034190, 0.0033980
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022446, 0.0022584

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017600, upper bound: 0.0017534
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017493, upper bound: 0.0017677
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012493, 0.0012671
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031703, 0.0032155
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019669, 0.0019949
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037250, 0.0036727
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032248, 0.0032707
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012215, 0.0012389
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047275, 0.0046611
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033081, 0.0032616
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035468, 0.0034970
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023100, 0.0023429

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017753, upper bound: 0.0017284
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017753, upper bound: 0.0017285
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012618, 0.0012551
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0032020, 0.0031850
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019865, 0.0019760
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036896, 0.0037093
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032570, 0.0032396
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012336, 0.0012271
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046826, 0.0047076
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032767, 0.0032942
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035131, 0.0035319
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023330, 0.0023206

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018270, upper bound: 0.0018186
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018381, upper bound: 0.0018080
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012449, 0.0012502
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031592, 0.0031725
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019600, 0.0019682
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036752, 0.0036597
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032134, 0.0032270
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012171, 0.0012223
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046643, 0.0046447
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032638, 0.0032501
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034994, 0.0034846
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023018, 0.0023115

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016982, upper bound: 0.0016647
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016974, upper bound: 0.0016646
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012449, 0.0012579
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031590, 0.0031920
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019599, 0.0019803
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036978, 0.0036596
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032133, 0.0032468
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012171, 0.0012298
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046929, 0.0046445
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032839, 0.0032500
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035208, 0.0034845
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023017, 0.0023257

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017037, upper bound: 0.0016801
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017141, upper bound: 0.0016695
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012414, 0.0012481
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031501, 0.0031672
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019544, 0.0019649
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036690, 0.0036493
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032042, 0.0032215
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012137, 0.0012202
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046564, 0.0046314
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032584, 0.0032408
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034935, 0.0034747
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022952, 0.0023076

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015044, upper bound: 0.0014693
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015044, upper bound: 0.0014693
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012416, 0.0012479
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031508, 0.0031666
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019548, 0.0019646
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036684, 0.0036500
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032049, 0.0032210
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012139, 0.0012200
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046557, 0.0046323
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032578, 0.0032415
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034929, 0.0034754
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022957, 0.0023073

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017700, upper bound: 0.0017601
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017846, upper bound: 0.0017463
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012324, 0.0012487
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031274, 0.0031687
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019403, 0.0019659
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036708, 0.0036230
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031811, 0.0032231
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012049, 0.0012208
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046587, 0.0045980
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032599, 0.0032175
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034951, 0.0034497
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022787, 0.0023087

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016061, upper bound: 0.0015742
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016062, upper bound: 0.0015741
time: 1.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012418, 0.0012467
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031511, 0.0031636
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019550, 0.0019627
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036648, 0.0036504
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032052, 0.0032179
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012141, 0.0012188
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046512, 0.0046329
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032547, 0.0032419
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034895, 0.0034758
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022960, 0.0023050

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013123, upper bound: 0.0012973
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013123, upper bound: 0.0012973
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012021, 0.0011869
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030505, 0.0030119
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018925, 0.0018686
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034891, 0.0035338
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031029, 0.0030636
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011753, 0.0011604
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044281, 0.0044849
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030986, 0.0031383
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033222, 0.0033648
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022226, 0.0021945

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 250

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016300, upper bound: 0.0017120
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016307, upper bound: 0.0017138
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012007, 0.0011842
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030470, 0.0030050
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018904, 0.0018643
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034811, 0.0035298
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030993, 0.0030566
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011739, 0.0011578
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044180, 0.0044798
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030915, 0.0031348
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033146, 0.0033610
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022201, 0.0021895

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016348, upper bound: 0.0017057
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016353, upper bound: 0.0017065
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012000, 0.0011941
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030452, 0.0030302
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018893, 0.0018800
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035103, 0.0035278
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030975, 0.0030822
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011733, 0.0011675
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044551, 0.0044772
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031175, 0.0031329
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033424, 0.0033590
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022188, 0.0022078

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016300, upper bound: 0.0017120
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016307, upper bound: 0.0017138
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012008, 0.0011914
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030473, 0.0030233
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018906, 0.0018757
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035024, 0.0035301
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030996, 0.0030752
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011740, 0.0011648
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044450, 0.0044802
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031104, 0.0031350
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033348, 0.0033612
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022203, 0.0022028

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016348, upper bound: 0.0017057
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016353, upper bound: 0.0017065
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011426, 0.0011218
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028996, 0.0028466
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017989, 0.0017661
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032977, 0.0033590
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029494, 0.0028955
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011171, 0.0010967
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041852, 0.0042630
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029286, 0.0029831
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031399, 0.0031983
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021127, 0.0020741

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015822, upper bound: 0.0016852
time: 1.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015822, upper bound: 0.0016852
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011577, 0.0011085
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029377, 0.0028129
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018226, 0.0017451
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032586, 0.0034032
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029882, 0.0028612
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011318, 0.0010837
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041356, 0.0043191
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028939, 0.0030223
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031027, 0.0032404
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021405, 0.0020495

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0016854
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0016854
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012466, 0.0012002
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031635, 0.0030457
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019627, 0.0018896
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035284, 0.0036648
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032178, 0.0030980
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012188, 0.0011735
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044779, 0.0046511
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031334, 0.0032546
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033595, 0.0034895
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023050, 0.0022192

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016896, upper bound: 0.0017954
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016896, upper bound: 0.0017955
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012471, 0.0011993
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031647, 0.0030435
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019634, 0.0018882
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035257, 0.0036662
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0032190, 0.0030957
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012193, 0.0011726
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044746, 0.0046528
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031311, 0.0032558
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033570, 0.0034908
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0023059, 0.0022175

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014339, upper bound: 0.0015092
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014339, upper bound: 0.0015092
time: 1.14 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 7.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017127, upper bound: 0.0015917
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016851, upper bound: 0.0016170
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0018107, upper bound: 0.0017241
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0018199, upper bound: 0.0017168
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016512, upper bound: 0.0015677
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016511, upper bound: 0.0015673
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0014902, upper bound: 0.0014396
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0014902, upper bound: 0.0014396
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016709, upper bound: 0.0015769
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016480, upper bound: 0.0015841
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0015035
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0015035
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0018058, upper bound: 0.0017060
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0018047, upper bound: 0.0017055
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017839, upper bound: 0.0016824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017842, upper bound: 0.0016821
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016107, upper bound: 0.0016226
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015916, upper bound: 0.0016454
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016955, upper bound: 0.0017374
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016955, upper bound: 0.0017362
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017332, upper bound: 0.0017746
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017332, upper bound: 0.0017750
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017228, upper bound: 0.0017703
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017245, upper bound: 0.0017647
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016973, upper bound: 0.0017051
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016998, upper bound: 0.0017023
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016232, upper bound: 0.0016030
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015974, upper bound: 0.0016319
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017601, upper bound: 0.0017700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017606, upper bound: 0.0017716
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017600, upper bound: 0.0017534
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017493, upper bound: 0.0017677
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017753, upper bound: 0.0017284
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017753, upper bound: 0.0017285
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0018270, upper bound: 0.0018186
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0018381, upper bound: 0.0018080
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016982, upper bound: 0.0016647
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016974, upper bound: 0.0016646
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017037, upper bound: 0.0016801
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017141, upper bound: 0.0016695
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015044, upper bound: 0.0014693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015044, upper bound: 0.0014693
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017700, upper bound: 0.0017601
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0017846, upper bound: 0.0017463
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016061, upper bound: 0.0015742
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016062, upper bound: 0.0015741
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0013123, upper bound: 0.0012973
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0013123, upper bound: 0.0012973
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016300, upper bound: 0.0017120
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016307, upper bound: 0.0017138
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016348, upper bound: 0.0017057
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016353, upper bound: 0.0017065
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016300, upper bound: 0.0017120
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016307, upper bound: 0.0017138
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016348, upper bound: 0.0017057
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016353, upper bound: 0.0017065
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015822, upper bound: 0.0016852
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015822, upper bound: 0.0016852
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0016854
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0015836, upper bound: 0.0016854
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016896, upper bound: 0.0017954
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0016896, upper bound: 0.0017955
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0014339, upper bound: 0.0015092
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.05
Output dim: 7, lower bound: -0.0014339, upper bound: 0.0015092

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010572, 0.0011171
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026828, 0.0028349
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016644, 0.0017588
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032841, 0.0031079
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027289, 0.0028836
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010336, 0.0010922
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041679, 0.0039443
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029165, 0.0027601
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031270, 0.0029592
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019547, 0.0020655

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016648, upper bound: 0.0015489
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016660, upper bound: 0.0015455
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010939, 0.0010981
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027760, 0.0027866
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017223, 0.0017288
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032282, 0.0032159
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028237, 0.0028345
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010695, 0.0010736
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040970, 0.0040814
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028669, 0.0028560
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030737, 0.0030620
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020226, 0.0020304

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011828, upper bound: 0.0011491
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011828, upper bound: 0.0011491
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010771, 0.0010866
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027333, 0.0027575
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016958, 0.0017108
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031944, 0.0031664
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027802, 0.0028048
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010531, 0.0010624
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040541, 0.0040186
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028369, 0.0028120
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030416, 0.0030149
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019915, 0.0020091

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016178, upper bound: 0.0015410
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016178, upper bound: 0.0015410
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010741, 0.0010915
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027256, 0.0027698
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016910, 0.0017184
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032086, 0.0031575
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027724, 0.0028173
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010501, 0.0010671
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040722, 0.0040073
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028495, 0.0028041
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030551, 0.0030065
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019859, 0.0020181

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016733, upper bound: 0.0015700
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016733, upper bound: 0.0015700
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010875, 0.0011278
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027597, 0.0028620
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017121, 0.0017756
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033155, 0.0031970
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028071, 0.0029111
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010632, 0.0011026
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042077, 0.0040574
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029444, 0.0028392
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031568, 0.0030440
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020107, 0.0020853

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014966, upper bound: 0.0014259
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014966, upper bound: 0.0014259
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010985, 0.0011122
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027876, 0.0028224
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017294, 0.0017511
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032697, 0.0032292
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028354, 0.0028709
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010740, 0.0010874
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041496, 0.0040983
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029037, 0.0028678
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031132, 0.0030747
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020310, 0.0020565

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013249, upper bound: 0.0012725
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0013249, upper bound: 0.0012725
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011626, 0.0011972
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029501, 0.0030382
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018303, 0.0018849
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035196, 0.0034176
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030008, 0.0030903
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011366, 0.0011705
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044668, 0.0043374
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031257, 0.0030351
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033512, 0.0032541
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021495, 0.0022137

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016514, upper bound: 0.0015563
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016505, upper bound: 0.0015557
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011721, 0.0011842
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029745, 0.0030052
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018454, 0.0018644
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034814, 0.0034458
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030256, 0.0030568
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011460, 0.0011578
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044183, 0.0043732
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030917, 0.0030601
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033148, 0.0032809
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021672, 0.0021896

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016059, upper bound: 0.0015436
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016059, upper bound: 0.0015436
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010543, 0.0010820
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026755, 0.0027457
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016599, 0.0017035
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031808, 0.0030995
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027215, 0.0027929
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010308, 0.0010579
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040369, 0.0039336
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028248, 0.0027526
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030286, 0.0029512
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019494, 0.0020006

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016630, upper bound: 0.0015590
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016630, upper bound: 0.0015590
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010671, 0.0010717
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027079, 0.0027196
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016800, 0.0016872
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031505, 0.0031370
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027544, 0.0027663
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010433, 0.0010478
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039984, 0.0039813
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027979, 0.0027859
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029998, 0.0029869
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019730, 0.0019815

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017511, upper bound: 0.0016512
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017512, upper bound: 0.0016508
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011538, 0.0011701
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029280, 0.0029692
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018165, 0.0018421
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034397, 0.0033919
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029782, 0.0030202
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011281, 0.0011440
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043654, 0.0043048
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030547, 0.0030123
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032751, 0.0032296
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021334, 0.0021634

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 112

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015881, upper bound: 0.0014936
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015881, upper bound: 0.0014936
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011548, 0.0011701
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029304, 0.0029694
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018180, 0.0018422
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034399, 0.0033948
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029807, 0.0030204
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011290, 0.0011440
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043657, 0.0043084
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030549, 0.0030148
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032754, 0.0032323
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021351, 0.0021636

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017691, upper bound: 0.0016599
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017486, upper bound: 0.0016656
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010416, 0.0010693
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026432, 0.0027135
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016398, 0.0016835
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031435, 0.0030620
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026886, 0.0027601
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010184, 0.0010455
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039895, 0.0038861
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027917, 0.0027193
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029931, 0.0029155
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019259, 0.0019771

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015117, upper bound: 0.0015275
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015117, upper bound: 0.0015275
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010787, 0.0010506
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027373, 0.0026660
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016982, 0.0016540
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030884, 0.0031710
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027843, 0.0027117
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010546, 0.0010271
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039196, 0.0040245
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027427, 0.0028161
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029406, 0.0030193
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019944, 0.0019425

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015480, upper bound: 0.0016006
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015486, upper bound: 0.0015980
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010577, 0.0010403
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026841, 0.0026398
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016652, 0.0016377
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030581, 0.0031094
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027302, 0.0026851
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010341, 0.0010171
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038811, 0.0039462
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027158, 0.0027614
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029118, 0.0029606
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019557, 0.0019234

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015120, upper bound: 0.0015284
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015020, upper bound: 0.0015371
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010586, 0.0010395
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026864, 0.0026379
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016667, 0.0016366
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030559, 0.0031121
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027325, 0.0026832
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010350, 0.0010163
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038783, 0.0039497
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027138, 0.0027638
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029097, 0.0029632
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019574, 0.0019220

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014104, upper bound: 0.0014375
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014104, upper bound: 0.0014375
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010856, 0.0010820
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027550, 0.0027457
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017092, 0.0017035
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031808, 0.0031915
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028023, 0.0027929
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010614, 0.0010579
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040368, 0.0040504
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028248, 0.0028343
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030286, 0.0030388
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020073, 0.0020006

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017175, upper bound: 0.0017456
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017075, upper bound: 0.0017578
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011003, 0.0010692
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027921, 0.0027133
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017322, 0.0016833
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031432, 0.0032345
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028400, 0.0027599
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010757, 0.0010454
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039891, 0.0041050
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027914, 0.0028725
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029928, 0.0030797
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020343, 0.0019769

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016741, upper bound: 0.0017184
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016741, upper bound: 0.0017171
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011676, 0.0011485
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029629, 0.0029144
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018382, 0.0018081
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033763, 0.0034324
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030138, 0.0029645
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011415, 0.0011229
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042849, 0.0043561
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029984, 0.0030482
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032147, 0.0032681
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021588, 0.0021235

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017073, upper bound: 0.0017412
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016968, upper bound: 0.0017537
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011665, 0.0011468
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029601, 0.0029102
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018364, 0.0018055
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0033713, 0.0034291
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030109, 0.0029602
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011404, 0.0011212
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0042787, 0.0043520
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029940, 0.0030453
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032100, 0.0032650
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021567, 0.0021204

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015701, upper bound: 0.0016122
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015701, upper bound: 0.0016122
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010336, 0.0010426
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026229, 0.0026457
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016272, 0.0016414
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030650, 0.0030385
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026679, 0.0026912
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010105, 0.0010193
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038898, 0.0038562
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027219, 0.0026984
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029183, 0.0028931
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019111, 0.0019277

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016756
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016682, upper bound: 0.0016892
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010273, 0.0010427
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026069, 0.0026459
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016173, 0.0016415
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030651, 0.0030200
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026517, 0.0026913
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010044, 0.0010194
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038901, 0.0038327
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027221, 0.0026820
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029185, 0.0028755
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0018994, 0.0019278

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016851, upper bound: 0.0016909
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016856, upper bound: 0.0016923
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010538, 0.0010618
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026742, 0.0026944
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016591, 0.0016716
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031213, 0.0030979
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027201, 0.0027406
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010303, 0.0010381
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039614, 0.0039317
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027720, 0.0027512
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029720, 0.0029497
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019485, 0.0019632

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0016097, upper bound: 0.0015714
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015985, upper bound: 0.0015855
time: 1.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010910, 0.0010439
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027686, 0.0026490
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017177, 0.0016435
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030687, 0.0032073
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028162, 0.0026945
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010667, 0.0010206
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038946, 0.0040705
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027253, 0.0028484
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029219, 0.0030539
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020173, 0.0019301

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015767, upper bound: 0.0016172
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015775, upper bound: 0.0016191
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012110, 0.0012192
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030730, 0.0030938
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019065, 0.0019194
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035840, 0.0035599
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031257, 0.0031469
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011839, 0.0011920
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045486, 0.0045180
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031829, 0.0031615
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034126, 0.0033896
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022390, 0.0022542

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017434, upper bound: 0.0017363
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017309, upper bound: 0.0017517
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012115, 0.0012189
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030743, 0.0030931
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019073, 0.0019190
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035832, 0.0035614
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031271, 0.0031462
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011844, 0.0011917
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045475, 0.0045199
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031821, 0.0031628
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034117, 0.0033910
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022400, 0.0022536

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015723, upper bound: 0.0015971
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015723, upper bound: 0.0015971
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012135, 0.0012331
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030793, 0.0031292
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019104, 0.0019414
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036250, 0.0035673
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031322, 0.0031829
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011864, 0.0012056
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046006, 0.0045273
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032193, 0.0031680
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034516, 0.0033966
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022437, 0.0022800

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0017363
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017417, upper bound: 0.0017384
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012256, 0.0012205
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031100, 0.0030973
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019295, 0.0019216
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035881, 0.0036028
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031634, 0.0031505
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011982, 0.0011933
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045537, 0.0045725
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031865, 0.0031996
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034164, 0.0034305
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022660, 0.0022567

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016909, upper bound: 0.0017129
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016968, upper bound: 0.0017114
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012289, 0.0012517
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031186, 0.0031764
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019348, 0.0019706
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0036797, 0.0036127
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031721, 0.0032309
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012015, 0.0012238
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0046700, 0.0045850
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0032678, 0.0032084
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035036, 0.0034399
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022722, 0.0023143

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017544, upper bound: 0.0017083
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017544, upper bound: 0.0017083
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012339, 0.0012671
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031312, 0.0032155
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019426, 0.0019949
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0037250, 0.0036273
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031850, 0.0032707
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012064, 0.0012389
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0047275, 0.0046036
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0033081, 0.0032214
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0035468, 0.0034538
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022814, 0.0023429

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014932, upper bound: 0.0014494
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014932, upper bound: 0.0014494
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012301, 0.0012198
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031215, 0.0030955
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019366, 0.0019204
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035859, 0.0036161
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031751, 0.0031486
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012026, 0.0011926
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045510, 0.0045893
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031846, 0.0032114
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034144, 0.0034431
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022744, 0.0022554

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017794, upper bound: 0.0017674
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017805, upper bound: 0.0017676
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012265, 0.0012229
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031125, 0.0031032
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019310, 0.0019252
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035949, 0.0036056
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031659, 0.0031565
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011992, 0.0011956
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045624, 0.0045760
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031926, 0.0032021
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034229, 0.0034331
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022678, 0.0022610

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016897, upper bound: 0.0016564
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016897, upper bound: 0.0016564
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011056, 0.0011222
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028057, 0.0028478
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017407, 0.0017668
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032990, 0.0032503
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028539, 0.0028967
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010810, 0.0010972
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041869, 0.0041250
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029298, 0.0028865
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031412, 0.0030948
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020443, 0.0020749

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016563, upper bound: 0.0016268
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016613, upper bound: 0.0016242
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011170, 0.0011059
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028345, 0.0028063
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017585, 0.0017410
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032510, 0.0032836
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028831, 0.0028545
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010920, 0.0010812
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041259, 0.0041673
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028871, 0.0029161
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030954, 0.0031265
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020652, 0.0020447

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016168
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016538, upper bound: 0.0016170
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012113, 0.0012208
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030740, 0.0030979
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019071, 0.0019219
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035888, 0.0035610
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031267, 0.0031511
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011843, 0.0011935
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045546, 0.0045194
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031871, 0.0031625
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034171, 0.0033907
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022397, 0.0022572

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Candidate
type: DSZ, layer: 1, pos: 176

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016587, upper bound: 0.0016320
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016609, upper bound: 0.0016322
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012079, 0.0012239
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030653, 0.0031057
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019017, 0.0019268
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035978, 0.0035510
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031179, 0.0031590
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011810, 0.0011966
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045661, 0.0045067
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031951, 0.0031536
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0034257, 0.0033811
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022334, 0.0022629

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016770, upper bound: 0.0016321
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016766, upper bound: 0.0016323
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012083, 0.0012110
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030663, 0.0030730
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019023, 0.0019065
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035599, 0.0035521
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031189, 0.0031257
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011814, 0.0011839
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045180, 0.0045081
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031615, 0.0031546
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033896, 0.0033822
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022341, 0.0022390

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017517, upper bound: 0.0017309
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017364, upper bound: 0.0017434
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012047, 0.0012141
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030571, 0.0030811
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018966, 0.0019115
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035693, 0.0035415
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031096, 0.0031340
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011778, 0.0011871
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0045299, 0.0044947
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031698, 0.0031451
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033985, 0.0033721
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022275, 0.0022449

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017324, upper bound: 0.0016942
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016942
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012026, 0.0011876
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030519, 0.0030138
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018934, 0.0018698
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034913, 0.0035354
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031043, 0.0030655
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011758, 0.0011611
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044309, 0.0044869
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031006, 0.0031397
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033243, 0.0033663
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022236, 0.0021959

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015936, upper bound: 0.0016753
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016753
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012028, 0.0011874
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030523, 0.0030133
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018937, 0.0018694
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034907, 0.0035360
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031047, 0.0030650
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011760, 0.0011609
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044302, 0.0044876
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031000, 0.0031402
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033237, 0.0033668
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022240, 0.0021955

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016767
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015947, upper bound: 0.0016769
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012013, 0.0011848
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030484, 0.0030067
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018912, 0.0018654
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034831, 0.0035314
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031007, 0.0030583
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011745, 0.0011584
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044205, 0.0044819
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030933, 0.0031362
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033165, 0.0033625
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022211, 0.0021907

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015974, upper bound: 0.0016663
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015992, upper bound: 0.0016679
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012014, 0.0011847
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030488, 0.0030064
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018915, 0.0018652
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034827, 0.0035319
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031011, 0.0030580
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011746, 0.0011583
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044201, 0.0044824
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030929, 0.0031366
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033161, 0.0033629
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022214, 0.0021905

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016591
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0016588
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012006, 0.0011949
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030466, 0.0030321
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018901, 0.0018812
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035126, 0.0035294
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030989, 0.0030842
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011738, 0.0011682
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044579, 0.0044792
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031195, 0.0031343
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033445, 0.0033605
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022198, 0.0022093

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015936, upper bound: 0.0016752
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016753
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012008, 0.0011947
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030473, 0.0030316
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018905, 0.0018808
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035120, 0.0035301
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030996, 0.0030837
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011740, 0.0011680
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044572, 0.0044802
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031189, 0.0031350
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033440, 0.0033612
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022203, 0.0022089

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015385, upper bound: 0.0016127
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015385, upper bound: 0.0016127
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012014, 0.0011921
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030487, 0.0030251
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018914, 0.0018768
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035044, 0.0035317
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031010, 0.0030770
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011746, 0.0011655
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044475, 0.0044822
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031122, 0.0031365
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033367, 0.0033628
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022213, 0.0022041

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015449, upper bound: 0.0016000
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015449, upper bound: 0.0016000
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012016, 0.0011920
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0030493, 0.0030247
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018918, 0.0018766
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035040, 0.0035325
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031016, 0.0030767
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011748, 0.0011654
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044471, 0.0044831
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031118, 0.0031371
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033364, 0.0033635
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022218, 0.0022039

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016591
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0016588
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011241, 0.0011075
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028526, 0.0028103
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017698, 0.0017435
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032556, 0.0033046
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029016, 0.0028586
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010990, 0.0010828
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041318, 0.0041940
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028912, 0.0029348
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030999, 0.0031465
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020785, 0.0020476

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011306, upper bound: 0.0011688
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011306, upper bound: 0.0011688
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011283, 0.0011218
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028633, 0.0028466
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017764, 0.0017661
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032977, 0.0033170
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029124, 0.0028955
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011032, 0.0010967
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041852, 0.0042097
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0029286, 0.0029457
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031399, 0.0031583
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020862, 0.0020741

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015617, upper bound: 0.0016651
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015619, upper bound: 0.0016655
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011384, 0.0010942
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028889, 0.0027766
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017923, 0.0017226
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032165, 0.0033466
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029385, 0.0028243
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011130, 0.0010698
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040822, 0.0042473
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028565, 0.0029720
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030627, 0.0031865
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021049, 0.0020231

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015428, upper bound: 0.0016452
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015428, upper bound: 0.0016452
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011434, 0.0011085
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029014, 0.0028129
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018001, 0.0017451
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032586, 0.0033612
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029513, 0.0028612
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011179, 0.0010837
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041356, 0.0042658
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028939, 0.0029850
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0031027, 0.0032004
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021140, 0.0020495

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015389, upper bound: 0.0016508
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0016432
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012307, 0.0011864
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031231, 0.0030107
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019376, 0.0018679
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034878, 0.0036180
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031767, 0.0030624
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012033, 0.0011600
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044265, 0.0045917
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030974, 0.0032130
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033209, 0.0034449
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022755, 0.0021937

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016467, upper bound: 0.0017587
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016543, upper bound: 0.0017546
time: 1.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0012328, 0.0011855
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0031285, 0.0030084
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0019409, 0.0018664
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034851, 0.0036242
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0031822, 0.0030601
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0012053, 0.0011591
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044231, 0.0045996
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030951, 0.0032186
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033184, 0.0034508
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0022795, 0.0021920

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016554, upper bound: 0.0017590
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016577, upper bound: 0.0017617
time: 2.19 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016648, upper bound: 0.0015489
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016660, upper bound: 0.0015455
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0011828, upper bound: 0.0011491
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0011828, upper bound: 0.0011491
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016178, upper bound: 0.0015410
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016178, upper bound: 0.0015410
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016733, upper bound: 0.0015700
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016733, upper bound: 0.0015700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0014966, upper bound: 0.0014259
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0014966, upper bound: 0.0014259
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0013249, upper bound: 0.0012725
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0013249, upper bound: 0.0012725
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016514, upper bound: 0.0015563
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016505, upper bound: 0.0015557
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016059, upper bound: 0.0015436
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016059, upper bound: 0.0015436
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016630, upper bound: 0.0015590
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016630, upper bound: 0.0015590
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017511, upper bound: 0.0016512
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017512, upper bound: 0.0016508
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015881, upper bound: 0.0014936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015881, upper bound: 0.0014936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017691, upper bound: 0.0016599
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017486, upper bound: 0.0016656
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015117, upper bound: 0.0015275
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015117, upper bound: 0.0015275
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015480, upper bound: 0.0016006
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015486, upper bound: 0.0015980
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015120, upper bound: 0.0015284
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015020, upper bound: 0.0015371
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0014104, upper bound: 0.0014375
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0014104, upper bound: 0.0014375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017175, upper bound: 0.0017456
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017075, upper bound: 0.0017578
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016741, upper bound: 0.0017184
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016741, upper bound: 0.0017171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017073, upper bound: 0.0017412
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016968, upper bound: 0.0017537
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015701, upper bound: 0.0016122
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015701, upper bound: 0.0016122
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016756
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016682, upper bound: 0.0016892
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016851, upper bound: 0.0016909
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016856, upper bound: 0.0016923
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016097, upper bound: 0.0015714
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015985, upper bound: 0.0015855
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015767, upper bound: 0.0016172
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015775, upper bound: 0.0016191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017434, upper bound: 0.0017363
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017309, upper bound: 0.0017517
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015723, upper bound: 0.0015971
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015723, upper bound: 0.0015971
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0017363
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017417, upper bound: 0.0017384
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016909, upper bound: 0.0017129
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016968, upper bound: 0.0017114
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017544, upper bound: 0.0017083
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017544, upper bound: 0.0017083
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0014932, upper bound: 0.0014494
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0014932, upper bound: 0.0014494
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017794, upper bound: 0.0017674
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017805, upper bound: 0.0017676
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016897, upper bound: 0.0016564
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016897, upper bound: 0.0016564
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016563, upper bound: 0.0016268
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016613, upper bound: 0.0016242
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016168
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016538, upper bound: 0.0016170
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016587, upper bound: 0.0016320
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016609, upper bound: 0.0016322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016770, upper bound: 0.0016321
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016766, upper bound: 0.0016323
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017517, upper bound: 0.0017309
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017364, upper bound: 0.0017434
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017324, upper bound: 0.0016942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015936, upper bound: 0.0016753
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016753
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016767
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015947, upper bound: 0.0016769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015974, upper bound: 0.0016663
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015992, upper bound: 0.0016679
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016591
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0016588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015936, upper bound: 0.0016752
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016753
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015385, upper bound: 0.0016127
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015385, upper bound: 0.0016127
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015449, upper bound: 0.0016000
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015449, upper bound: 0.0016000
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016591
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0016588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0011306, upper bound: 0.0011688
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0011306, upper bound: 0.0011688
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015617, upper bound: 0.0016651
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015619, upper bound: 0.0016655
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015428, upper bound: 0.0016452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015428, upper bound: 0.0016452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015389, upper bound: 0.0016508
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0016432
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016467, upper bound: 0.0017587
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016543, upper bound: 0.0017546
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016554, upper bound: 0.0017590
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.75
Output dim: 7, lower bound: -0.0016577, upper bound: 0.0017617

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010122, 0.0010674
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0025686, 0.0027088
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0015936, 0.0016805
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031380, 0.0029756
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026127, 0.0027553
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0009896, 0.0010436
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039825, 0.0037764
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027868, 0.0026425
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029878, 0.0028332
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0018715, 0.0019736

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011409, upper bound: 0.0011120
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011409, upper bound: 0.0011120
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010075, 0.0010651
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0025567, 0.0027029
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0015862, 0.0016769
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031312, 0.0029618
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026006, 0.0027493
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0009850, 0.0010414
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039739, 0.0037589
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027807, 0.0026303
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029814, 0.0028201
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0018628, 0.0019694

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010569, 0.0010790
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026820, 0.0027380
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016639, 0.0016987
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031719, 0.0031069
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027280, 0.0027850
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010333, 0.0010549
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040255, 0.0039431
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028169, 0.0027592
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030201, 0.0029583
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019541, 0.0019950

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016537, upper bound: 0.0015499
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0015499
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010616, 0.0010915
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026939, 0.0027698
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016713, 0.0017184
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032086, 0.0031208
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0027402, 0.0028173
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010379, 0.0010671
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040722, 0.0039607
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028495, 0.0027715
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030551, 0.0029715
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019628, 0.0020181

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016299, upper bound: 0.0015339
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016393, upper bound: 0.0015286
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011596, 0.0011945
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029426, 0.0030313
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018256, 0.0018806
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035116, 0.0034089
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029931, 0.0030834
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011337, 0.0011679
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044567, 0.0043263
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031186, 0.0030273
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033436, 0.0032458
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021440, 0.0022087

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015509, upper bound: 0.0014639
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015509, upper bound: 0.0014639
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011598, 0.0011942
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029431, 0.0030304
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018259, 0.0018800
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0035105, 0.0034095
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029936, 0.0030824
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011339, 0.0011675
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0044553, 0.0043270
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0031176, 0.0030278
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0033426, 0.0032463
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021444, 0.0022080

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0014635
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0014635
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010402, 0.0010702
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026396, 0.0027159
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016376, 0.0016849
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031462, 0.0030578
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026849, 0.0027625
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010170, 0.0010464
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0039930, 0.0038807
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027941, 0.0027156
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029957, 0.0029115
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019232, 0.0019788

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0015389
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016429, upper bound: 0.0015389
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010426, 0.0010820
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026457, 0.0027457
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016414, 0.0017035
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0031808, 0.0030649
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026911, 0.0027929
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010193, 0.0010579
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040369, 0.0038897
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028248, 0.0027218
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030286, 0.0029182
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019277, 0.0020006

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010324, 0.0010378
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026197, 0.0026337
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016253, 0.0016339
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030510, 0.0030349
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026647, 0.0026789
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010093, 0.0010147
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038721, 0.0038516
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027095, 0.0026952
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029050, 0.0028897
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019088, 0.0019189

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017322, upper bound: 0.0016307
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017317, upper bound: 0.0016306
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010333, 0.0010370
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0026220, 0.0026316
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0016267, 0.0016326
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0030486, 0.0030375
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0026670, 0.0026768
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010102, 0.0010139
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0038690, 0.0038550
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0027073, 0.0026975
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0029027, 0.0028922
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0019104, 0.0019174

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017350, upper bound: 0.0016270
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017158, upper bound: 0.0016335
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011516, 0.0011789
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029224, 0.0029916
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018130, 0.0018560
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034657, 0.0033854
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0029725, 0.0030430
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011259, 0.0011526
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043984, 0.0042965
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030778, 0.0030065
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032999, 0.0032234
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021293, 0.0021797

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014480, upper bound: 0.0013734
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014480, upper bound: 0.0013734
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011632, 0.0011680
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0029518, 0.0029639
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0018313, 0.0018388
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0034335, 0.0034195
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0030025, 0.0030148
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0011373, 0.0011419
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0043576, 0.0043398
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0030492, 0.0030368
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0032692, 0.0032559
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0021507, 0.0021595

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014266, upper bound: 0.0013781
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0014266, upper bound: 0.0013781
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0010950, 0.0011051
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0027787, 0.0028042
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017239, 0.0017398
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032486, 0.0032190
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028264, 0.0028524
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010706, 0.0010804
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0041229, 0.0040853
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028850, 0.0028587
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030932, 0.0030650
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020246, 0.0020432

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015641, upper bound: 0.0016001
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0015641, upper bound: 0.0016001
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0011068, 0.0010926
1: -0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0028088, 0.0027727
2: 0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0017426, 0.0017202
3: 0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0032120, 0.0032538
4: -0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0028570, 0.0028203
5: 0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0010821, 0.0010682
6: 0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0040764, 0.0041295
7: 0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0028525, 0.0028896
8: -0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0030583, 0.0030981
9: -0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0020465, 0.0020202

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016648, upper bound: 0.0017205
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016676, upper bound: 0.0017116
time: 2.01 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0011409, upper bound: 0.0011120
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0011409, upper bound: 0.0011120
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016537, upper bound: 0.0015499
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016534, upper bound: 0.0015499
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016299, upper bound: 0.0015339
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016393, upper bound: 0.0015286
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0015509, upper bound: 0.0014639
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0015509, upper bound: 0.0014639
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0014635
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0014635
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016432, upper bound: 0.0015389
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016429, upper bound: 0.0015389
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0011450, upper bound: 0.0011069
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0017322, upper bound: 0.0016307
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0017317, upper bound: 0.0016306
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0017350, upper bound: 0.0016270
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0017158, upper bound: 0.0016335
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0014480, upper bound: 0.0013734
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0014480, upper bound: 0.0013734
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0014266, upper bound: 0.0013781
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0014266, upper bound: 0.0013781
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0015641, upper bound: 0.0016001
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0015641, upper bound: 0.0016001
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016648, upper bound: 0.0017205
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.28
Output dim: 7, lower bound: -0.0016676, upper bound: 0.0017116
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016741, upper bound: 0.0017184
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016741, upper bound: 0.0017171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017073, upper bound: 0.0017412
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016968, upper bound: 0.0017537
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016827, upper bound: 0.0016756
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016682, upper bound: 0.0016892
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016851, upper bound: 0.0016909
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016856, upper bound: 0.0016923
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015775, upper bound: 0.0016191
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017434, upper bound: 0.0017363
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017309, upper bound: 0.0017517
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017411, upper bound: 0.0017363
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017417, upper bound: 0.0017384
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016909, upper bound: 0.0017129
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016968, upper bound: 0.0017114
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017544, upper bound: 0.0017083
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017544, upper bound: 0.0017083
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017794, upper bound: 0.0017674
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017805, upper bound: 0.0017676
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016897, upper bound: 0.0016564
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016897, upper bound: 0.0016564
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016563, upper bound: 0.0016268
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016613, upper bound: 0.0016242
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016168
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016538, upper bound: 0.0016170
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016587, upper bound: 0.0016320
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016609, upper bound: 0.0016322
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016770, upper bound: 0.0016321
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016766, upper bound: 0.0016323
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017517, upper bound: 0.0017309
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017364, upper bound: 0.0017434
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017324, upper bound: 0.0016942
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0017344, upper bound: 0.0016942
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015936, upper bound: 0.0016753
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016753
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016767
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015947, upper bound: 0.0016769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015974, upper bound: 0.0016663
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015992, upper bound: 0.0016679
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016591
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0016588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015936, upper bound: 0.0016752
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015940, upper bound: 0.0016753
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015898, upper bound: 0.0016591
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015902, upper bound: 0.0016588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015617, upper bound: 0.0016651
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015619, upper bound: 0.0016655
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015428, upper bound: 0.0016452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015428, upper bound: 0.0016452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015389, upper bound: 0.0016508
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0015505, upper bound: 0.0016432
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016467, upper bound: 0.0017587
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016543, upper bound: 0.0017546
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016554, upper bound: 0.0017590
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 7, lower bound: -0.0016577, upper bound: 0.0017617

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.58 + 599.54 = 603.12 seconds
