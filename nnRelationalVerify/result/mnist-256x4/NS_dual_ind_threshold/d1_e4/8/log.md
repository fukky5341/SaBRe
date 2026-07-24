## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0015192279999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823)
1: (-0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038504, 0.0038504)
2: (0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026485)
3: (-0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0046414, 0.0046414)
4: (-0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040695, 0.0040695)
5: (0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447)
6: (-0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056837, 0.0056837)
7: (0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0039407, 0.0039407)
8: (-0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088)
9: (0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.60 + 2.73 = 4.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015303, upper bound: 0.0015812
time: 1.84 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015812
time: 1.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.72
Output dim: 2, lower bound: -0.0015303, upper bound: 0.0015812
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.72
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015812

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0008951, 0.0005296, -0.0009888, 0.0006195, -0.0012381, 0.0015184
1: -0.0065819, -0.0029667, -0.0068197, -0.0027386, -0.0031418, 0.0033231
2: 0.0309466, 0.0331895, 0.0307990, 0.0333310, -0.0019492, 0.0023143
3: -0.0024107, 0.0017774, -0.0026749, 0.0020529, -0.0040052, 0.0036396
4: -0.0055879, -0.0019106, -0.0058298, -0.0012397, -0.0035666, 0.0031276
5: 0.0116216, 0.0130145, 0.0115300, 0.0131024, -0.0012104, 0.0014845
6: -0.0026714, 0.0026439, -0.0034461, 0.0029935, -0.0045207, 0.0049876
7: 0.9761900, 0.9799094, 0.9759553, 0.9801540, -0.0034013, 0.0032322
8: -0.0120923, -0.0081046, -0.0123439, -0.0078423, -0.0042500, 0.0034655
9: 0.0003539, 0.0029881, 0.0001807, 0.0031542, -0.0022891, 0.0028074

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014605, upper bound: 0.0014996
time: 1.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014689, upper bound: 0.0015194
time: 2.02 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0010127, 0.0006190, -0.0010389, 0.0006253, -0.0016379, 0.0016579
1: -0.0068804, -0.0027398, -0.0069470, -0.0027239, -0.0033099, 0.0037837
2: 0.0307614, 0.0333303, 0.0307200, 0.0333401, -0.0023445, 0.0025988
3: -0.0026735, 0.0021231, -0.0026920, 0.0022004, -0.0045442, 0.0040211
4: -0.0058915, -0.0012447, -0.0059593, -0.0011761, -0.0034817, 0.0039577
5: 0.0115066, 0.0131019, 0.0114810, 0.0131080, -0.0016014, 0.0016210
6: -0.0034405, 0.0030827, -0.0035164, 0.0031807, -0.0055544, 0.0048512
7: 0.9759565, 0.9802164, 0.9759402, 0.9802849, -0.0038742, 0.0033844
8: -0.0123426, -0.0077754, -0.0123601, -0.0077019, -0.0046407, 0.0045847
9: 0.0001365, 0.0031534, 0.0000879, 0.0031650, -0.0030285, 0.0030654

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015303
time: 1.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015812
time: 1.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.59 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 5.59
Output dim: 2, lower bound: -0.0014605, upper bound: 0.0014996
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 2, lower bound: -0.0014689, upper bound: 0.0015194
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015303
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015812

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0008948, 0.0005063, -0.0009879, 0.0005429, -0.0011265, 0.0011939
1: -0.0065813, -0.0030258, -0.0068175, -0.0029330, -0.0028587, 0.0030296
2: 0.0309470, 0.0331528, 0.0308004, 0.0332104, -0.0017735, 0.0018796
3: -0.0023421, 0.0017767, -0.0024497, 0.0020503, -0.0035097, 0.0033116
4: -0.0055873, -0.0019708, -0.0058276, -0.0018763, -0.0029077, 0.0030817
5: 0.0116219, 0.0129917, 0.0115309, 0.0130275, -0.0011014, 0.0011672
6: -0.0025843, 0.0026429, -0.0027209, 0.0029903, -0.0044543, 0.0042029
7: 0.9762509, 0.9799086, 0.9761553, 0.9801517, -0.0031169, 0.0029410
8: -0.0120270, -0.0081053, -0.0121295, -0.0078447, -0.0033418, 0.0031532
9: 0.0003544, 0.0029449, 0.0001823, 0.0030126, -0.0020829, 0.0022074

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013975, upper bound: 0.0014589
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014084, upper bound: 0.0014576
time: 1.83 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0010127, 0.0006190, -0.0008951, 0.0005296, -0.0015423, 0.0012372
1: -0.0068804, -0.0027398, -0.0065819, -0.0029667, -0.0034026, 0.0031396
2: 0.0307614, 0.0333303, 0.0309466, 0.0331895, -0.0023078, 0.0019478
3: -0.0026735, 0.0021231, -0.0024107, 0.0017774, -0.0036371, 0.0040686
4: -0.0058915, -0.0012447, -0.0055879, -0.0019106, -0.0032465, 0.0035617
5: 0.0115066, 0.0131019, 0.0116216, 0.0130145, -0.0015079, 0.0012096
6: -0.0034405, 0.0030827, -0.0026714, 0.0026439, -0.0049820, 0.0046925
7: 0.9759565, 0.9802164, 0.9761900, 0.9799094, -0.0032300, 0.0034856
8: -0.0123426, -0.0077754, -0.0120923, -0.0081046, -0.0034631, 0.0043169
9: 0.0001365, 0.0031534, 0.0003539, 0.0029881, -0.0028515, 0.0022876

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014995, upper bound: 0.0014604
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015195, upper bound: 0.0014689
time: 2.09 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0010127, 0.0006190, -0.0010127, 0.0006190, -0.0016317, 0.0016317
1: -0.0068804, -0.0027398, -0.0068804, -0.0027398, -0.0032985, 0.0032985
2: 0.0307614, 0.0333303, 0.0307614, 0.0333303, -0.0023375, 0.0023375
3: -0.0026735, 0.0021231, -0.0026735, 0.0021231, -0.0040080, 0.0040080
4: -0.0058915, -0.0012447, -0.0058915, -0.0012447, -0.0034251, 0.0034251
5: 0.0115066, 0.0131019, 0.0115066, 0.0131019, -0.0015953, 0.0015953
6: -0.0034405, 0.0030827, -0.0034405, 0.0030827, -0.0047901, 0.0047901
7: 0.9759565, 0.9802164, 0.9759565, 0.9802164, -0.0033727, 0.0033727
8: -0.0123426, -0.0077754, -0.0123426, -0.0077754, -0.0045672, 0.0045672
9: 0.0001365, 0.0031534, 0.0001365, 0.0031534, -0.0030169, 0.0030169

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014996, upper bound: 0.0014604
time: 1.95 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015194, upper bound: 0.0014689
time: 2.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.78 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 5.78
Output dim: 2, lower bound: -0.0013975, upper bound: 0.0014589
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 5.78
Output dim: 2, lower bound: -0.0014084, upper bound: 0.0014576
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.78
Output dim: 2, lower bound: -0.0014995, upper bound: 0.0014604
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.78
Output dim: 2, lower bound: -0.0015195, upper bound: 0.0014689
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 5.78
Output dim: 2, lower bound: -0.0014996, upper bound: 0.0014604
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.78
Output dim: 2, lower bound: -0.0015194, upper bound: 0.0014689

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0010118, 0.0005423, -0.0008948, 0.0005063, -0.0012400, 0.0011256
1: -0.0068781, -0.0029343, -0.0065813, -0.0030258, -0.0031466, 0.0028564
2: 0.0307628, 0.0332096, 0.0309470, 0.0331528, -0.0019521, 0.0017721
3: -0.0024482, 0.0021206, -0.0023421, 0.0017767, -0.0033090, 0.0036452
4: -0.0058892, -0.0018777, -0.0055873, -0.0019708, -0.0032006, 0.0029054
5: 0.0115075, 0.0130270, 0.0116219, 0.0129917, -0.0012123, 0.0011005
6: -0.0027189, 0.0030794, -0.0025843, 0.0026429, -0.0041995, 0.0046262
7: 0.9761568, 0.9802141, 0.9762509, 0.9799086, -0.0029386, 0.0032372
8: -0.0121280, -0.0077779, -0.0120270, -0.0081053, -0.0031507, 0.0034708
9: 0.0001381, 0.0030116, 0.0003544, 0.0029449, -0.0022926, 0.0020812

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014589, upper bound: 0.0013975
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014578, upper bound: 0.0014084
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0010118, 0.0005423, -0.0010124, 0.0005946, -0.0011702, 0.0014541
1: -0.0068781, -0.0029343, -0.0068797, -0.0028017, -0.0029695, 0.0028673
2: 0.0307628, 0.0332096, 0.0307619, 0.0332918, -0.0018423, 0.0019001
3: -0.0024482, 0.0021206, -0.0026018, 0.0021224, -0.0033992, 0.0034400
4: -0.0058892, -0.0018777, -0.0058908, -0.0015117, -0.0032029, 0.0027821
5: 0.0115075, 0.0130270, 0.0115069, 0.0130780, -0.0011441, 0.0013998
6: -0.0027189, 0.0030794, -0.0031452, 0.0030817, -0.0040213, 0.0045474
7: 0.9761568, 0.9802141, 0.9760203, 0.9802157, -0.0029406, 0.0030550
8: -0.0121280, -0.0077779, -0.0122743, -0.0077761, -0.0039755, 0.0032754
9: 0.0001381, 0.0030116, 0.0001370, 0.0031083, -0.0021636, 0.0025283

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014589, upper bound: 0.0013974
time: 2.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014578, upper bound: 0.0014084
time: 1.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.20 seconds
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 2, lower bound: -0.0014589, upper bound: 0.0013975
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 2, lower bound: -0.0014578, upper bound: 0.0014084
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 2, lower bound: -0.0014589, upper bound: 0.0013974
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 2, lower bound: -0.0014578, upper bound: 0.0014084

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.33 + 41.64 = 45.97 seconds
