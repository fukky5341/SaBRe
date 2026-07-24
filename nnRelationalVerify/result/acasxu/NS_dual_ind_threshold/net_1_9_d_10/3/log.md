## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00038128


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0201701, -0.0196344, -0.0201701, -0.0196344, -0.0005356, 0.0005356)
1: (-0.0186188, -0.0174458, -0.0186188, -0.0174458, -0.0011730, 0.0011730)
2: (-0.0187211, -0.0175092, -0.0187211, -0.0175092, -0.0012119, 0.0012119)
3: (-0.0178066, -0.0164318, -0.0178066, -0.0164318, -0.0013747, 0.0013747)
4: (-0.0178369, -0.0166623, -0.0178369, -0.0166623, -0.0011746, 0.0011746)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.14 + 0.55 = 1.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0004766, upper bound: 0.0004766

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002052, upper bound: 0.0004755
time: 0.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004761, upper bound: 0.0004761
time: 0.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.46 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.46
Output dim: 0, lower bound: -0.0002052, upper bound: 0.0004755
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.46
Output dim: 0, lower bound: -0.0004761, upper bound: 0.0004761

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0200573, -0.0196402, -0.0201701, -0.0196344, -0.0004229, 0.0005298
1: -0.0185711, -0.0175256, -0.0186188, -0.0174458, -0.0011253, 0.0010932
2: -0.0186060, -0.0175360, -0.0187211, -0.0175092, -0.0010968, 0.0011851
3: -0.0177788, -0.0165006, -0.0178066, -0.0164318, -0.0013470, 0.0013059
4: -0.0177621, -0.0166911, -0.0178369, -0.0166623, -0.0010999, 0.0011458

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002046, upper bound: 0.0002046
time: 0.16 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002046, upper bound: 0.0002046
time: 0.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0201701, -0.0196344, -0.0005354, 0.0005355
1: -0.0186186, -0.0174470, -0.0186188, -0.0174458, -0.0011728, 0.0011718
2: -0.0187207, -0.0175095, -0.0187211, -0.0175092, -0.0012114, 0.0012116
3: -0.0178064, -0.0164329, -0.0178066, -0.0164318, -0.0013745, 0.0013737
4: -0.0178364, -0.0166625, -0.0178369, -0.0166623, -0.0011741, 0.0011743

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052
time: 0.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052
time: 0.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.37 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 1.37
Output dim: 0, lower bound: -0.0002046, upper bound: 0.0002046
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 1.37
Output dim: 0, lower bound: -0.0002046, upper bound: 0.0002046
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.37
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.37
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0200573, -0.0196402, -0.0005297, 0.0004228
1: -0.0186186, -0.0174470, -0.0185711, -0.0175256, -0.0010930, 0.0011241
2: -0.0187207, -0.0175095, -0.0186060, -0.0175360, -0.0011846, 0.0010965
3: -0.0178064, -0.0164329, -0.0177788, -0.0165006, -0.0013058, 0.0013459
4: -0.0178364, -0.0166625, -0.0177621, -0.0166911, -0.0011453, 0.0010996

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001945
time: 0.15 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0002019
time: 0.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0201699, -0.0196345, -0.0005353, 0.0005353
1: -0.0186186, -0.0174470, -0.0186186, -0.0174470, -0.0011716, 0.0011716
2: -0.0187207, -0.0175095, -0.0187207, -0.0175095, -0.0012112, 0.0012112
3: -0.0178064, -0.0164329, -0.0178064, -0.0164329, -0.0013735, 0.0013735
4: -0.0178364, -0.0166625, -0.0178364, -0.0166625, -0.0011739, 0.0011739

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0002956
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0003303
time: 0.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.40 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001945
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0002019
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0002956
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.40
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0003303

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0200573, -0.0196402, -0.0005219, 0.0004162
1: -0.0186205, -0.0174573, -0.0185711, -0.0175256, -0.0010949, 0.0011138
2: -0.0186928, -0.0175274, -0.0186060, -0.0175360, -0.0011568, 0.0010786
3: -0.0178194, -0.0164468, -0.0177788, -0.0165006, -0.0013188, 0.0013321
4: -0.0178173, -0.0166758, -0.0177621, -0.0166911, -0.0011262, 0.0010864

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001900
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001945
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0200573, -0.0196402, -0.0005251, 0.0004212
1: -0.0186108, -0.0174589, -0.0185711, -0.0175256, -0.0010852, 0.0011121
2: -0.0186998, -0.0175154, -0.0186060, -0.0175360, -0.0011638, 0.0010906
3: -0.0177972, -0.0164446, -0.0177788, -0.0165006, -0.0012966, 0.0013342
4: -0.0178163, -0.0166688, -0.0177621, -0.0166911, -0.0011252, 0.0010933

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004616, upper bound: 0.0001973
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004616, upper bound: 0.0002019
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0201699, -0.0196345, -0.0005275, 0.0005287
1: -0.0186205, -0.0174573, -0.0186186, -0.0174470, -0.0011735, 0.0011613
2: -0.0186928, -0.0175274, -0.0187207, -0.0175095, -0.0011833, 0.0011932
3: -0.0178194, -0.0164468, -0.0178064, -0.0164329, -0.0013865, 0.0013596
4: -0.0178173, -0.0166758, -0.0178364, -0.0166625, -0.0011547, 0.0011606

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004395, upper bound: 0.0002956
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004395, upper bound: 0.0002956
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0201699, -0.0196345, -0.0005308, 0.0005338
1: -0.0186108, -0.0174589, -0.0186186, -0.0174470, -0.0011638, 0.0011597
2: -0.0186998, -0.0175154, -0.0187207, -0.0175095, -0.0011903, 0.0012052
3: -0.0177972, -0.0164446, -0.0178064, -0.0164329, -0.0013643, 0.0013618
4: -0.0178163, -0.0166688, -0.0178364, -0.0166625, -0.0011537, 0.0011676

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0003302
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0003303
time: 0.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.43 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001900
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001945
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004616, upper bound: 0.0001973
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004616, upper bound: 0.0002019
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004395, upper bound: 0.0002956
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004395, upper bound: 0.0002956
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0003302
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0004670, upper bound: 0.0003303

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0200512, -0.0196471, -0.0005150, 0.0004101
1: -0.0186205, -0.0174573, -0.0185725, -0.0175298, -0.0010907, 0.0011152
2: -0.0186928, -0.0175274, -0.0185882, -0.0175426, -0.0011502, 0.0010608
3: -0.0178194, -0.0164468, -0.0177999, -0.0165085, -0.0013109, 0.0013531
4: -0.0178173, -0.0166758, -0.0177291, -0.0166914, -0.0011258, 0.0010533

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004218, upper bound: 0.0000822
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004028, upper bound: 0.0001764
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0200546, -0.0196417, -0.0005204, 0.0004134
1: -0.0186205, -0.0174573, -0.0185629, -0.0175361, -0.0010844, 0.0011056
2: -0.0186928, -0.0175274, -0.0185846, -0.0175404, -0.0011524, 0.0010572
3: -0.0178194, -0.0164468, -0.0177695, -0.0165109, -0.0013084, 0.0013227
4: -0.0178173, -0.0166758, -0.0177360, -0.0166955, -0.0011218, 0.0010603

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004218, upper bound: 0.0000867
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004028, upper bound: 0.0001810
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0200512, -0.0196471, -0.0005183, 0.0004151
1: -0.0186108, -0.0174589, -0.0185725, -0.0175298, -0.0010810, 0.0011135
2: -0.0186998, -0.0175154, -0.0185882, -0.0175426, -0.0011573, 0.0010728
3: -0.0177972, -0.0164446, -0.0177999, -0.0165085, -0.0012887, 0.0013553
4: -0.0178163, -0.0166688, -0.0177291, -0.0166914, -0.0011249, 0.0010603

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0000887
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0001916
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0200546, -0.0196417, -0.0005237, 0.0004185
1: -0.0186108, -0.0174589, -0.0185629, -0.0175361, -0.0010747, 0.0011040
2: -0.0186998, -0.0175154, -0.0185846, -0.0175404, -0.0011595, 0.0010692
3: -0.0177972, -0.0164446, -0.0177695, -0.0165109, -0.0012863, 0.0013249
4: -0.0178163, -0.0166688, -0.0177360, -0.0166955, -0.0011208, 0.0010673

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0000933
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0001962
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0201621, -0.0196412, -0.0005209, 0.0005209
1: -0.0186205, -0.0174573, -0.0186205, -0.0174573, -0.0011632, 0.0011632
2: -0.0186928, -0.0175274, -0.0186928, -0.0175274, -0.0011654, 0.0011654
3: -0.0178194, -0.0164468, -0.0178194, -0.0164468, -0.0013726, 0.0013726
4: -0.0178173, -0.0166758, -0.0178173, -0.0166758, -0.0011415, 0.0011415

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004260, upper bound: 0.0002563
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004071, upper bound: 0.0002796
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0201653, -0.0196361, -0.0005260, 0.0005242
1: -0.0186205, -0.0174573, -0.0186108, -0.0174589, -0.0011616, 0.0011535
2: -0.0186928, -0.0175274, -0.0186998, -0.0175154, -0.0011774, 0.0011724
3: -0.0178194, -0.0164468, -0.0177972, -0.0164446, -0.0013748, 0.0013504
4: -0.0178173, -0.0166758, -0.0178163, -0.0166688, -0.0011485, 0.0011405

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004260, upper bound: 0.0002573
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004071, upper bound: 0.0002796
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0201621, -0.0196412, -0.0005242, 0.0005260
1: -0.0186108, -0.0174589, -0.0186205, -0.0174573, -0.0011535, 0.0011616
2: -0.0186998, -0.0175154, -0.0186928, -0.0175274, -0.0011724, 0.0011774
3: -0.0177972, -0.0164446, -0.0178194, -0.0164468, -0.0013504, 0.0013748
4: -0.0178163, -0.0166688, -0.0178173, -0.0166758, -0.0011405, 0.0011485

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0002782
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0003268
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0201653, -0.0196361, -0.0005292, 0.0005292
1: -0.0186108, -0.0174589, -0.0186108, -0.0174589, -0.0011518, 0.0011518
2: -0.0186998, -0.0175154, -0.0186998, -0.0175154, -0.0011844, 0.0011844
3: -0.0177972, -0.0164446, -0.0177972, -0.0164446, -0.0013526, 0.0013526
4: -0.0178163, -0.0166688, -0.0178163, -0.0166688, -0.0011475, 0.0011475

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0002840
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0003268
time: 0.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.55 seconds
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004218, upper bound: 0.0000822
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004028, upper bound: 0.0001764
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004218, upper bound: 0.0000867
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004028, upper bound: 0.0001810
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0000887
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0001916
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0000933
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004492, upper bound: 0.0001962
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004260, upper bound: 0.0002563
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004071, upper bound: 0.0002796
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004260, upper bound: 0.0002573
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004071, upper bound: 0.0002796
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0002782
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0003268
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0002840
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.55
Output dim: 0, lower bound: -0.0004534, upper bound: 0.0003268

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0200512, -0.0196471, -0.0005164, 0.0003441
1: -0.0186202, -0.0174565, -0.0185725, -0.0175298, -0.0010905, 0.0011160
2: -0.0187094, -0.0176630, -0.0185882, -0.0175426, -0.0011668, 0.0009252
3: -0.0177892, -0.0164457, -0.0177999, -0.0165085, -0.0012807, 0.0013542
4: -0.0179050, -0.0168030, -0.0177291, -0.0166914, -0.0012136, 0.0009261

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004043, upper bound: 0.0000167
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004206, upper bound: 0.0000736
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004191, upper bound: 0.0000768
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0200512, -0.0196471, -0.0005141, 0.0004054
1: -0.0186195, -0.0174784, -0.0185725, -0.0175298, -0.0010897, 0.0010941
2: -0.0186864, -0.0175455, -0.0185882, -0.0175426, -0.0011439, 0.0010428
3: -0.0178169, -0.0164651, -0.0177999, -0.0165085, -0.0013084, 0.0013348
4: -0.0178100, -0.0166962, -0.0177291, -0.0166914, -0.0011186, 0.0010329

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003854, upper bound: 0.0001428
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002296, upper bound: 0.0001080
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004016, upper bound: 0.0001068
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004001, upper bound: 0.0001478
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0200546, -0.0196417, -0.0005218, 0.0003475
1: -0.0186202, -0.0174565, -0.0185629, -0.0175361, -0.0010842, 0.0011064
2: -0.0187094, -0.0176630, -0.0185846, -0.0175404, -0.0011690, 0.0009216
3: -0.0177892, -0.0164457, -0.0177695, -0.0165109, -0.0012783, 0.0013238
4: -0.0179050, -0.0168030, -0.0177360, -0.0166955, -0.0012095, 0.0009331

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0200546, -0.0196417, -0.0005195, 0.0004087
1: -0.0186195, -0.0174784, -0.0185629, -0.0175361, -0.0010834, 0.0010845
2: -0.0186864, -0.0175455, -0.0185846, -0.0175404, -0.0011460, 0.0010391
3: -0.0178169, -0.0164651, -0.0177695, -0.0165109, -0.0013060, 0.0013044
4: -0.0178100, -0.0166962, -0.0177360, -0.0166955, -0.0011145, 0.0010398

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0200512, -0.0196471, -0.0005378, 0.0003486
1: -0.0186559, -0.0174389, -0.0185725, -0.0175298, -0.0011262, 0.0011335
2: -0.0188510, -0.0176227, -0.0185882, -0.0175426, -0.0013084, 0.0009655
3: -0.0178288, -0.0164218, -0.0177999, -0.0165085, -0.0013203, 0.0013781
4: -0.0180366, -0.0167791, -0.0177291, -0.0166914, -0.0013452, 0.0009500

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004318, upper bound: 0.0000371
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004480, upper bound: 0.0000794
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004467, upper bound: 0.0000887
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0200512, -0.0196471, -0.0005171, 0.0004088
1: -0.0186081, -0.0174854, -0.0185725, -0.0175298, -0.0010783, 0.0010870
2: -0.0186916, -0.0175474, -0.0185882, -0.0175426, -0.0011491, 0.0010409
3: -0.0177927, -0.0164679, -0.0177999, -0.0165085, -0.0012842, 0.0013320
4: -0.0178068, -0.0166938, -0.0177291, -0.0166914, -0.0011154, 0.0010353

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004318, upper bound: 0.0001657
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003590, upper bound: 0.0001534
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004480, upper bound: 0.0001275
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004467, upper bound: 0.0001916
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0200546, -0.0196417, -0.0005432, 0.0003520
1: -0.0186559, -0.0174389, -0.0185629, -0.0175361, -0.0011199, 0.0011240
2: -0.0188510, -0.0176227, -0.0185846, -0.0175404, -0.0013106, 0.0009618
3: -0.0178288, -0.0164218, -0.0177695, -0.0165109, -0.0013178, 0.0013477
4: -0.0180366, -0.0167791, -0.0177360, -0.0166955, -0.0013412, 0.0009570

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0200546, -0.0196417, -0.0005224, 0.0004122
1: -0.0186081, -0.0174854, -0.0185629, -0.0175361, -0.0010720, 0.0010775
2: -0.0186916, -0.0175474, -0.0185846, -0.0175404, -0.0011512, 0.0010372
3: -0.0177927, -0.0164679, -0.0177695, -0.0165109, -0.0012818, 0.0013016
4: -0.0178068, -0.0166938, -0.0177360, -0.0166955, -0.0011113, 0.0010422

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0201621, -0.0196412, -0.0005223, 0.0004550
1: -0.0186202, -0.0174565, -0.0186205, -0.0174573, -0.0011629, 0.0011640
2: -0.0187094, -0.0176630, -0.0186928, -0.0175274, -0.0011820, 0.0010298
3: -0.0177892, -0.0164457, -0.0178194, -0.0164468, -0.0013424, 0.0013737
4: -0.0179050, -0.0168030, -0.0178173, -0.0166758, -0.0012293, 0.0010143

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002567
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002567
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201621, -0.0196412, -0.0005200, 0.0005162
1: -0.0186195, -0.0174784, -0.0186205, -0.0174573, -0.0011622, 0.0011421
2: -0.0186864, -0.0175455, -0.0186928, -0.0175274, -0.0011590, 0.0011473
3: -0.0178169, -0.0164651, -0.0178194, -0.0164468, -0.0013702, 0.0013542
4: -0.0178100, -0.0166962, -0.0178173, -0.0166758, -0.0011343, 0.0011210

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002809
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002809
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0201653, -0.0196361, -0.0005274, 0.0004583
1: -0.0186202, -0.0174565, -0.0186108, -0.0174589, -0.0011613, 0.0011543
2: -0.0187094, -0.0176630, -0.0186998, -0.0175154, -0.0011940, 0.0010368
3: -0.0177892, -0.0164457, -0.0177972, -0.0164446, -0.0013446, 0.0013515
4: -0.0179050, -0.0168030, -0.0178163, -0.0166688, -0.0012362, 0.0010133

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201653, -0.0196361, -0.0005250, 0.0005195
1: -0.0186195, -0.0174784, -0.0186108, -0.0174589, -0.0011606, 0.0011324
2: -0.0186864, -0.0175455, -0.0186998, -0.0175154, -0.0011710, 0.0011544
3: -0.0178169, -0.0164651, -0.0177972, -0.0164446, -0.0013723, 0.0013321
4: -0.0178100, -0.0166962, -0.0178163, -0.0166688, -0.0011412, 0.0011201

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0201621, -0.0196412, -0.0005437, 0.0004595
1: -0.0186559, -0.0174389, -0.0186205, -0.0174573, -0.0011986, 0.0011816
2: -0.0188510, -0.0176227, -0.0186928, -0.0175274, -0.0013236, 0.0010701
3: -0.0178288, -0.0164218, -0.0178194, -0.0164468, -0.0013820, 0.0013976
4: -0.0180366, -0.0167791, -0.0178173, -0.0166758, -0.0013609, 0.0010382

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0002782
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0002782
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0201621, -0.0196412, -0.0005230, 0.0005197
1: -0.0186081, -0.0174854, -0.0186205, -0.0174573, -0.0011508, 0.0011351
2: -0.0186916, -0.0175474, -0.0186928, -0.0175274, -0.0011642, 0.0011454
3: -0.0177927, -0.0164679, -0.0178194, -0.0164468, -0.0013460, 0.0013515
4: -0.0178068, -0.0166938, -0.0178173, -0.0166758, -0.0011310, 0.0011235

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0003278
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0003278
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0201653, -0.0196361, -0.0005487, 0.0004627
1: -0.0186559, -0.0174389, -0.0186108, -0.0174589, -0.0011970, 0.0011718
2: -0.0188510, -0.0176227, -0.0186998, -0.0175154, -0.0013356, 0.0010771
3: -0.0178288, -0.0164218, -0.0177972, -0.0164446, -0.0013842, 0.0013754
4: -0.0180366, -0.0167791, -0.0178163, -0.0166688, -0.0013679, 0.0010372

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0201653, -0.0196361, -0.0005280, 0.0005229
1: -0.0186081, -0.0174854, -0.0186108, -0.0174589, -0.0011492, 0.0011253
2: -0.0186916, -0.0175474, -0.0186998, -0.0175154, -0.0011762, 0.0011524
3: -0.0177927, -0.0164679, -0.0177972, -0.0164446, -0.0013481, 0.0013293
4: -0.0178068, -0.0166938, -0.0178163, -0.0166688, -0.0011380, 0.0011225

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268
time: 0.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.58 seconds
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002567
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002567
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002809
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003128, upper bound: 0.0002809
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0002782
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0002782
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0003278
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003592, upper bound: 0.0003278
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.58
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0200946, -0.0196850, -0.0004785, 0.0003876
1: -0.0186202, -0.0174565, -0.0186069, -0.0176012, -0.0010191, 0.0011505
2: -0.0187094, -0.0176630, -0.0185809, -0.0175803, -0.0011291, 0.0009179
3: -0.0177892, -0.0164457, -0.0177866, -0.0165612, -0.0012280, 0.0013409
4: -0.0179050, -0.0168030, -0.0177143, -0.0167369, -0.0011681, 0.0009113

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0200405, -0.0196715, -0.0004920, 0.0003335
1: -0.0186202, -0.0174565, -0.0185575, -0.0175971, -0.0010231, 0.0011010
2: -0.0187094, -0.0176630, -0.0185536, -0.0175797, -0.0011297, 0.0008906
3: -0.0177892, -0.0164457, -0.0177534, -0.0165643, -0.0012249, 0.0013077
4: -0.0179050, -0.0168030, -0.0176978, -0.0167237, -0.0011814, 0.0008948

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0200946, -0.0196850, -0.0004762, 0.0004488
1: -0.0186195, -0.0174784, -0.0186069, -0.0176012, -0.0010183, 0.0011285
2: -0.0186864, -0.0175455, -0.0185809, -0.0175803, -0.0011061, 0.0010355
3: -0.0178169, -0.0164651, -0.0177866, -0.0165612, -0.0012557, 0.0013215
4: -0.0178100, -0.0166962, -0.0177143, -0.0167369, -0.0010731, 0.0010181

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0200405, -0.0196715, -0.0004896, 0.0003947
1: -0.0186195, -0.0174784, -0.0185575, -0.0175971, -0.0010224, 0.0010791
2: -0.0186864, -0.0175455, -0.0185536, -0.0175797, -0.0011067, 0.0010081
3: -0.0178169, -0.0164651, -0.0177534, -0.0165643, -0.0012526, 0.0012883
4: -0.0178100, -0.0166962, -0.0176978, -0.0167237, -0.0010864, 0.0010016

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.17 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0200946, -0.0196850, -0.0004999, 0.0003920
1: -0.0186559, -0.0174389, -0.0186069, -0.0176012, -0.0010548, 0.0011680
2: -0.0188510, -0.0176227, -0.0185809, -0.0175803, -0.0012707, 0.0009582
3: -0.0178288, -0.0164218, -0.0177866, -0.0165612, -0.0012675, 0.0013648
4: -0.0180366, -0.0167791, -0.0177143, -0.0167369, -0.0012997, 0.0009353

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0200405, -0.0196715, -0.0005133, 0.0003379
1: -0.0186559, -0.0174389, -0.0185575, -0.0175971, -0.0010589, 0.0011186
2: -0.0188510, -0.0176227, -0.0185536, -0.0175797, -0.0012713, 0.0009309
3: -0.0178288, -0.0164218, -0.0177534, -0.0165643, -0.0012645, 0.0013316
4: -0.0180366, -0.0167791, -0.0176978, -0.0167237, -0.0013130, 0.0009187

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0200946, -0.0196850, -0.0004792, 0.0004522
1: -0.0186081, -0.0174854, -0.0186069, -0.0176012, -0.0010070, 0.0011215
2: -0.0186916, -0.0175474, -0.0185809, -0.0175803, -0.0011113, 0.0010336
3: -0.0177927, -0.0164679, -0.0177866, -0.0165612, -0.0012315, 0.0013187
4: -0.0178068, -0.0166938, -0.0177143, -0.0167369, -0.0010698, 0.0010205

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001691
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0200405, -0.0196715, -0.0004926, 0.0003981
1: -0.0186081, -0.0174854, -0.0185575, -0.0175971, -0.0010110, 0.0010721
2: -0.0186916, -0.0175474, -0.0185536, -0.0175797, -0.0011119, 0.0010062
3: -0.0177927, -0.0164679, -0.0177534, -0.0165643, -0.0012284, 0.0012855
4: -0.0178068, -0.0166938, -0.0176978, -0.0167237, -0.0010831, 0.0010040

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001691
time: 0.18 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.57 seconds
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.57
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001691

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200405, -0.0196715, -0.0005394, 0.0003001
1: -0.0186439, -0.0175444, -0.0185575, -0.0175971, -0.0010468, 0.0010132
2: -0.0187389, -0.0176913, -0.0185536, -0.0175797, -0.0011592, 0.0008623
3: -0.0178018, -0.0165161, -0.0177534, -0.0165643, -0.0012375, 0.0012373
4: -0.0179121, -0.0168245, -0.0176978, -0.0167237, -0.0011885, 0.0008733

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200405, -0.0196715, -0.0004896, 0.0003021
1: -0.0186171, -0.0175008, -0.0185575, -0.0175971, -0.0010200, 0.0010567
2: -0.0186920, -0.0176950, -0.0185536, -0.0175797, -0.0011123, 0.0008586
3: -0.0177792, -0.0164873, -0.0177534, -0.0165643, -0.0012149, 0.0012661
4: -0.0178829, -0.0168297, -0.0176978, -0.0167237, -0.0011592, 0.0008681

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200405, -0.0196715, -0.0005494, 0.0003616
1: -0.0186431, -0.0175629, -0.0185575, -0.0175971, -0.0010460, 0.0009946
2: -0.0187184, -0.0175800, -0.0185536, -0.0175797, -0.0011386, 0.0009736
3: -0.0178262, -0.0165264, -0.0177534, -0.0165643, -0.0012619, 0.0012270
4: -0.0178274, -0.0167230, -0.0176978, -0.0167237, -0.0011037, 0.0009748

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200405, -0.0196715, -0.0004755, 0.0003609
1: -0.0186148, -0.0175297, -0.0185575, -0.0175971, -0.0010177, 0.0010278
2: -0.0186554, -0.0175794, -0.0185536, -0.0175797, -0.0010757, 0.0009742
3: -0.0178018, -0.0165113, -0.0177534, -0.0165643, -0.0012375, 0.0012421
4: -0.0177744, -0.0167240, -0.0176978, -0.0167237, -0.0010508, 0.0009738

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200946, -0.0196850, -0.0004968, 0.0003610
1: -0.0186521, -0.0174887, -0.0186069, -0.0176012, -0.0010510, 0.0011183
2: -0.0188297, -0.0176612, -0.0185809, -0.0175803, -0.0012494, 0.0009197
3: -0.0178194, -0.0164689, -0.0177866, -0.0165612, -0.0012582, 0.0013177
4: -0.0180115, -0.0168087, -0.0177143, -0.0167369, -0.0012745, 0.0009056

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200405, -0.0196715, -0.0005769, 0.0002962
1: -0.0186901, -0.0175541, -0.0185575, -0.0175971, -0.0010930, 0.0010034
2: -0.0188018, -0.0176594, -0.0185536, -0.0175797, -0.0012221, 0.0008942
3: -0.0178560, -0.0165166, -0.0177534, -0.0165643, -0.0012917, 0.0012368
4: -0.0179617, -0.0168128, -0.0176978, -0.0167237, -0.0012380, 0.0008850

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200405, -0.0196715, -0.0005103, 0.0003069
1: -0.0186521, -0.0174887, -0.0185575, -0.0175971, -0.0010551, 0.0010688
2: -0.0188297, -0.0176612, -0.0185536, -0.0175797, -0.0012500, 0.0008924
3: -0.0178194, -0.0164689, -0.0177534, -0.0165643, -0.0012551, 0.0012845
4: -0.0180115, -0.0168087, -0.0176978, -0.0167237, -0.0012878, 0.0008891

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200946, -0.0196850, -0.0004652, 0.0004219
1: -0.0186027, -0.0175448, -0.0186069, -0.0176012, -0.0010016, 0.0010621
2: -0.0186596, -0.0175877, -0.0185809, -0.0175803, -0.0010793, 0.0009933
3: -0.0177792, -0.0165207, -0.0177866, -0.0165612, -0.0012180, 0.0012659
4: -0.0177742, -0.0167241, -0.0177143, -0.0167369, -0.0010372, 0.0009902

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200405, -0.0196715, -0.0005598, 0.0003456
1: -0.0186527, -0.0175739, -0.0185575, -0.0175971, -0.0010556, 0.0009836
2: -0.0187115, -0.0175769, -0.0185536, -0.0175797, -0.0011318, 0.0009767
3: -0.0178239, -0.0165320, -0.0177534, -0.0165643, -0.0012596, 0.0012214
4: -0.0178280, -0.0167218, -0.0176978, -0.0167237, -0.0011044, 0.0009760

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200405, -0.0196715, -0.0004787, 0.0003678
1: -0.0186027, -0.0175448, -0.0185575, -0.0175971, -0.0010057, 0.0010127
2: -0.0186596, -0.0175877, -0.0185536, -0.0175797, -0.0010799, 0.0009659
3: -0.0177792, -0.0165207, -0.0177534, -0.0165643, -0.0012149, 0.0012327
4: -0.0177742, -0.0167241, -0.0176978, -0.0167237, -0.0010505, 0.0009737

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.19 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.65 seconds
NS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
NS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200405, -0.0196715, -0.0005394, 0.0003001
1: -0.0186439, -0.0175444, -0.0185575, -0.0175971, -0.0010468, 0.0010132
2: -0.0187389, -0.0176913, -0.0185536, -0.0175797, -0.0011592, 0.0008623
3: -0.0178018, -0.0165161, -0.0177534, -0.0165643, -0.0012375, 0.0012373
4: -0.0179121, -0.0168245, -0.0176978, -0.0167237, -0.0011885, 0.0008733

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000753
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200405, -0.0196715, -0.0004896, 0.0003021
1: -0.0186171, -0.0175008, -0.0185575, -0.0175971, -0.0010200, 0.0010567
2: -0.0186920, -0.0176950, -0.0185536, -0.0175797, -0.0011123, 0.0008586
3: -0.0177792, -0.0164873, -0.0177534, -0.0165643, -0.0012149, 0.0012661
4: -0.0178829, -0.0168297, -0.0176978, -0.0167237, -0.0011592, 0.0008681

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200405, -0.0196715, -0.0005494, 0.0003616
1: -0.0186431, -0.0175629, -0.0185575, -0.0175971, -0.0010460, 0.0009946
2: -0.0187184, -0.0175800, -0.0185536, -0.0175797, -0.0011386, 0.0009736
3: -0.0178262, -0.0165264, -0.0177534, -0.0165643, -0.0012619, 0.0012270
4: -0.0178274, -0.0167230, -0.0176978, -0.0167237, -0.0011037, 0.0009748

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001618
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200405, -0.0196715, -0.0004755, 0.0003609
1: -0.0186148, -0.0175297, -0.0185575, -0.0175971, -0.0010177, 0.0010278
2: -0.0186554, -0.0175794, -0.0185536, -0.0175797, -0.0010757, 0.0009742
3: -0.0178018, -0.0165113, -0.0177534, -0.0165643, -0.0012375, 0.0012421
4: -0.0177744, -0.0167240, -0.0176978, -0.0167237, -0.0010508, 0.0009738

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200946, -0.0196850, -0.0004968, 0.0003610
1: -0.0186521, -0.0174887, -0.0186069, -0.0176012, -0.0010510, 0.0011183
2: -0.0188297, -0.0176612, -0.0185809, -0.0175803, -0.0012494, 0.0009197
3: -0.0178194, -0.0164689, -0.0177866, -0.0165612, -0.0012582, 0.0013177
4: -0.0180115, -0.0168087, -0.0177143, -0.0167369, -0.0012745, 0.0009056

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200382, -0.0196719, -0.0005098, 0.0003045
1: -0.0186521, -0.0174887, -0.0185494, -0.0176000, -0.0010522, 0.0010608
2: -0.0188297, -0.0176612, -0.0185388, -0.0175805, -0.0012492, 0.0008775
3: -0.0178194, -0.0164689, -0.0177483, -0.0165663, -0.0012532, 0.0012794
4: -0.0180115, -0.0168087, -0.0176821, -0.0167244, -0.0012871, 0.0008735

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200405, -0.0196715, -0.0005769, 0.0002962
1: -0.0186901, -0.0175541, -0.0185575, -0.0175971, -0.0010930, 0.0010034
2: -0.0188018, -0.0176594, -0.0185536, -0.0175797, -0.0012221, 0.0008942
3: -0.0178560, -0.0165166, -0.0177534, -0.0165643, -0.0012917, 0.0012368
4: -0.0179617, -0.0168128, -0.0176978, -0.0167237, -0.0012380, 0.0008850

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200946, -0.0196850, -0.0004968, 0.0003610
1: -0.0186521, -0.0174887, -0.0186069, -0.0176012, -0.0010510, 0.0011183
2: -0.0188297, -0.0176612, -0.0185809, -0.0175803, -0.0012494, 0.0009197
3: -0.0178194, -0.0164689, -0.0177866, -0.0165612, -0.0012582, 0.0013177
4: -0.0180115, -0.0168087, -0.0177143, -0.0167369, -0.0012745, 0.0009056

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200405, -0.0196715, -0.0005103, 0.0003069
1: -0.0186521, -0.0174887, -0.0185575, -0.0175971, -0.0010551, 0.0010688
2: -0.0188297, -0.0176612, -0.0185536, -0.0175797, -0.0012500, 0.0008924
3: -0.0178194, -0.0164689, -0.0177534, -0.0165643, -0.0012551, 0.0012845
4: -0.0180115, -0.0168087, -0.0176978, -0.0167237, -0.0012878, 0.0008891

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200382, -0.0196719, -0.0005594, 0.0003433
1: -0.0186527, -0.0175739, -0.0185494, -0.0176000, -0.0010528, 0.0009755
2: -0.0187115, -0.0175769, -0.0185388, -0.0175805, -0.0011310, 0.0009619
3: -0.0178239, -0.0165320, -0.0177483, -0.0165663, -0.0012576, 0.0012162
4: -0.0178280, -0.0167218, -0.0176821, -0.0167244, -0.0011036, 0.0009603

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200946, -0.0196850, -0.0004652, 0.0004219
1: -0.0186027, -0.0175448, -0.0186069, -0.0176012, -0.0010016, 0.0010621
2: -0.0186596, -0.0175877, -0.0185809, -0.0175803, -0.0010793, 0.0009933
3: -0.0177792, -0.0165207, -0.0177866, -0.0165612, -0.0012180, 0.0012659
4: -0.0177742, -0.0167241, -0.0177143, -0.0167369, -0.0010372, 0.0009902

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200382, -0.0196719, -0.0004782, 0.0003655
1: -0.0186027, -0.0175448, -0.0185494, -0.0176000, -0.0010028, 0.0010046
2: -0.0186596, -0.0175877, -0.0185388, -0.0175805, -0.0010791, 0.0009511
3: -0.0177792, -0.0165207, -0.0177483, -0.0165663, -0.0012129, 0.0012276
4: -0.0177742, -0.0167241, -0.0176821, -0.0167244, -0.0010498, 0.0009580

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200405, -0.0196715, -0.0005598, 0.0003456
1: -0.0186527, -0.0175739, -0.0185575, -0.0175971, -0.0010556, 0.0009836
2: -0.0187115, -0.0175769, -0.0185536, -0.0175797, -0.0011318, 0.0009767
3: -0.0178239, -0.0165320, -0.0177534, -0.0165643, -0.0012596, 0.0012214
4: -0.0178280, -0.0167218, -0.0176978, -0.0167237, -0.0011044, 0.0009760

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200946, -0.0196850, -0.0004652, 0.0004219
1: -0.0186027, -0.0175448, -0.0186069, -0.0176012, -0.0010016, 0.0010621
2: -0.0186596, -0.0175877, -0.0185809, -0.0175803, -0.0010793, 0.0009933
3: -0.0177792, -0.0165207, -0.0177866, -0.0165612, -0.0012180, 0.0012659
4: -0.0177742, -0.0167241, -0.0177143, -0.0167369, -0.0010372, 0.0009902

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200405, -0.0196715, -0.0004787, 0.0003678
1: -0.0186027, -0.0175448, -0.0185575, -0.0175971, -0.0010057, 0.0010127
2: -0.0186596, -0.0175877, -0.0185536, -0.0175797, -0.0010799, 0.0009659
3: -0.0177792, -0.0165207, -0.0177534, -0.0165643, -0.0012149, 0.0012327
4: -0.0177742, -0.0167241, -0.0176978, -0.0167237, -0.0010505, 0.0009737

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.20 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 1.75 seconds
NS_A2_B1_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
NS_A2_B1_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
NS_A2_B1_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
NS_A2_B1_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000753
NS_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
NS_A2_B1_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
NS_A2_B1_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
NS_A2_B1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001618
NS_A2_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
NS_A2_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
NS_A2_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
NS_A2_B1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
NS_A2_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
NS_A2_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.75
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200405, -0.0196715, -0.0005394, 0.0003001
1: -0.0186439, -0.0175444, -0.0185575, -0.0175971, -0.0010468, 0.0010132
2: -0.0187389, -0.0176913, -0.0185536, -0.0175797, -0.0011592, 0.0008623
3: -0.0178018, -0.0165161, -0.0177534, -0.0165643, -0.0012375, 0.0012373
4: -0.0179121, -0.0168245, -0.0176978, -0.0167237, -0.0011885, 0.0008733

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200405, -0.0196715, -0.0004896, 0.0003021
1: -0.0186171, -0.0175008, -0.0185575, -0.0175971, -0.0010200, 0.0010567
2: -0.0186920, -0.0176950, -0.0185536, -0.0175797, -0.0011123, 0.0008586
3: -0.0177792, -0.0164873, -0.0177534, -0.0165643, -0.0012149, 0.0012661
4: -0.0178829, -0.0168297, -0.0176978, -0.0167237, -0.0011592, 0.0008681

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200405, -0.0196715, -0.0005394, 0.0003001
1: -0.0186439, -0.0175444, -0.0185575, -0.0175971, -0.0010468, 0.0010132
2: -0.0187389, -0.0176913, -0.0185536, -0.0175797, -0.0011592, 0.0008623
3: -0.0178018, -0.0165161, -0.0177534, -0.0165643, -0.0012375, 0.0012373
4: -0.0179121, -0.0168245, -0.0176978, -0.0167237, -0.0011885, 0.0008733

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200405, -0.0196715, -0.0004896, 0.0003021
1: -0.0186171, -0.0175008, -0.0185575, -0.0175971, -0.0010200, 0.0010567
2: -0.0186920, -0.0176950, -0.0185536, -0.0175797, -0.0011123, 0.0008586
3: -0.0177792, -0.0164873, -0.0177534, -0.0165643, -0.0012149, 0.0012661
4: -0.0178829, -0.0168297, -0.0176978, -0.0167237, -0.0011592, 0.0008681

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200405, -0.0196715, -0.0005494, 0.0003616
1: -0.0186431, -0.0175629, -0.0185575, -0.0175971, -0.0010460, 0.0009946
2: -0.0187184, -0.0175800, -0.0185536, -0.0175797, -0.0011386, 0.0009736
3: -0.0178262, -0.0165264, -0.0177534, -0.0165643, -0.0012619, 0.0012270
4: -0.0178274, -0.0167230, -0.0176978, -0.0167237, -0.0011037, 0.0009748

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200405, -0.0196715, -0.0004755, 0.0003609
1: -0.0186148, -0.0175297, -0.0185575, -0.0175971, -0.0010177, 0.0010278
2: -0.0186554, -0.0175794, -0.0185536, -0.0175797, -0.0010757, 0.0009742
3: -0.0178018, -0.0165113, -0.0177534, -0.0165643, -0.0012375, 0.0012421
4: -0.0177744, -0.0167240, -0.0176978, -0.0167237, -0.0010508, 0.0009738

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200405, -0.0196715, -0.0005494, 0.0003616
1: -0.0186431, -0.0175629, -0.0185575, -0.0175971, -0.0010460, 0.0009946
2: -0.0187184, -0.0175800, -0.0185536, -0.0175797, -0.0011386, 0.0009736
3: -0.0178262, -0.0165264, -0.0177534, -0.0165643, -0.0012619, 0.0012270
4: -0.0178274, -0.0167230, -0.0176978, -0.0167237, -0.0011037, 0.0009748

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200405, -0.0196715, -0.0004755, 0.0003609
1: -0.0186148, -0.0175297, -0.0185575, -0.0175971, -0.0010177, 0.0010278
2: -0.0186554, -0.0175794, -0.0185536, -0.0175797, -0.0010757, 0.0009742
3: -0.0178018, -0.0165113, -0.0177534, -0.0165643, -0.0012375, 0.0012421
4: -0.0177744, -0.0167240, -0.0176978, -0.0167237, -0.0010508, 0.0009738

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200946, -0.0196850, -0.0004893, 0.0003590
1: -0.0186353, -0.0175003, -0.0186069, -0.0176012, -0.0010342, 0.0011066
2: -0.0187326, -0.0176718, -0.0185809, -0.0175803, -0.0011522, 0.0009092
3: -0.0178025, -0.0164775, -0.0177866, -0.0165612, -0.0012413, 0.0013091
4: -0.0179169, -0.0168191, -0.0177143, -0.0167369, -0.0011799, 0.0008952

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200382, -0.0196719, -0.0005023, 0.0003026
1: -0.0186353, -0.0175003, -0.0185494, -0.0176000, -0.0010354, 0.0010491
2: -0.0187326, -0.0176718, -0.0185388, -0.0175805, -0.0011520, 0.0008670
3: -0.0178025, -0.0164775, -0.0177483, -0.0165663, -0.0012362, 0.0012707
4: -0.0179169, -0.0168191, -0.0176821, -0.0167244, -0.0011925, 0.0008630

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200946, -0.0196850, -0.0004968, 0.0003610
1: -0.0186521, -0.0174887, -0.0186069, -0.0176012, -0.0010510, 0.0011183
2: -0.0188297, -0.0176612, -0.0185809, -0.0175803, -0.0012494, 0.0009197
3: -0.0178194, -0.0164689, -0.0177866, -0.0165612, -0.0012582, 0.0013177
4: -0.0180115, -0.0168087, -0.0177143, -0.0167369, -0.0012745, 0.0009056

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200382, -0.0196719, -0.0005098, 0.0003045
1: -0.0186521, -0.0174887, -0.0185494, -0.0176000, -0.0010522, 0.0010608
2: -0.0188297, -0.0176612, -0.0185388, -0.0175805, -0.0012492, 0.0008775
3: -0.0178194, -0.0164689, -0.0177483, -0.0165663, -0.0012532, 0.0012794
4: -0.0180115, -0.0168087, -0.0176821, -0.0167244, -0.0012871, 0.0008735

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200946, -0.0196850, -0.0004893, 0.0003590
1: -0.0186353, -0.0175003, -0.0186069, -0.0176012, -0.0010342, 0.0011066
2: -0.0187326, -0.0176718, -0.0185809, -0.0175803, -0.0011522, 0.0009092
3: -0.0178025, -0.0164775, -0.0177866, -0.0165612, -0.0012413, 0.0013091
4: -0.0179169, -0.0168191, -0.0177143, -0.0167369, -0.0011799, 0.0008952

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200405, -0.0196715, -0.0005769, 0.0002962
1: -0.0186901, -0.0175541, -0.0185575, -0.0175971, -0.0010930, 0.0010034
2: -0.0188018, -0.0176594, -0.0185536, -0.0175797, -0.0012221, 0.0008942
3: -0.0178560, -0.0165166, -0.0177534, -0.0165643, -0.0012917, 0.0012368
4: -0.0179617, -0.0168128, -0.0176978, -0.0167237, -0.0012380, 0.0008850

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200405, -0.0196715, -0.0005027, 0.0003049
1: -0.0186353, -0.0175003, -0.0185575, -0.0175971, -0.0010383, 0.0010572
2: -0.0187326, -0.0176718, -0.0185536, -0.0175797, -0.0011528, 0.0008818
3: -0.0178025, -0.0164775, -0.0177534, -0.0165643, -0.0012382, 0.0012759
4: -0.0179169, -0.0168191, -0.0176978, -0.0167237, -0.0011932, 0.0008787

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200946, -0.0196850, -0.0004968, 0.0003610
1: -0.0186521, -0.0174887, -0.0186069, -0.0176012, -0.0010510, 0.0011183
2: -0.0188297, -0.0176612, -0.0185809, -0.0175803, -0.0012494, 0.0009197
3: -0.0178194, -0.0164689, -0.0177866, -0.0165612, -0.0012582, 0.0013177
4: -0.0180115, -0.0168087, -0.0177143, -0.0167369, -0.0012745, 0.0009056

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200405, -0.0196715, -0.0005769, 0.0002962
1: -0.0186901, -0.0175541, -0.0185575, -0.0175971, -0.0010930, 0.0010034
2: -0.0188018, -0.0176594, -0.0185536, -0.0175797, -0.0012221, 0.0008942
3: -0.0178560, -0.0165166, -0.0177534, -0.0165643, -0.0012917, 0.0012368
4: -0.0179617, -0.0168128, -0.0176978, -0.0167237, -0.0012380, 0.0008850

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200405, -0.0196715, -0.0005103, 0.0003069
1: -0.0186521, -0.0174887, -0.0185575, -0.0175971, -0.0010551, 0.0010688
2: -0.0188297, -0.0176612, -0.0185536, -0.0175797, -0.0012500, 0.0008924
3: -0.0178194, -0.0164689, -0.0177534, -0.0165643, -0.0012551, 0.0012845
4: -0.0180115, -0.0168087, -0.0176978, -0.0167237, -0.0012878, 0.0008891

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201467, -0.0196733, -0.0200946, -0.0196850, -0.0004618, 0.0004213
1: -0.0185973, -0.0175479, -0.0186069, -0.0176012, -0.0009961, 0.0010590
2: -0.0186375, -0.0175907, -0.0185809, -0.0175803, -0.0010572, 0.0009902
3: -0.0177729, -0.0165232, -0.0177866, -0.0165612, -0.0012117, 0.0012634
4: -0.0177583, -0.0167275, -0.0177143, -0.0167369, -0.0010213, 0.0009868

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200382, -0.0196719, -0.0005594, 0.0003433
1: -0.0186527, -0.0175739, -0.0185494, -0.0176000, -0.0010528, 0.0009755
2: -0.0187115, -0.0175769, -0.0185388, -0.0175805, -0.0011310, 0.0009619
3: -0.0178239, -0.0165320, -0.0177483, -0.0165663, -0.0012576, 0.0012162
4: -0.0178280, -0.0167218, -0.0176821, -0.0167244, -0.0011036, 0.0009603

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201467, -0.0196733, -0.0200382, -0.0196719, -0.0004748, 0.0003649
1: -0.0185973, -0.0175479, -0.0185494, -0.0176000, -0.0009973, 0.0010015
2: -0.0186375, -0.0175907, -0.0185388, -0.0175805, -0.0010570, 0.0009481
3: -0.0177729, -0.0165232, -0.0177483, -0.0165663, -0.0012066, 0.0012251
4: -0.0177583, -0.0167275, -0.0176821, -0.0167244, -0.0010339, 0.0009547

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200946, -0.0196850, -0.0004652, 0.0004219
1: -0.0186027, -0.0175448, -0.0186069, -0.0176012, -0.0010016, 0.0010621
2: -0.0186596, -0.0175877, -0.0185809, -0.0175803, -0.0010793, 0.0009933
3: -0.0177792, -0.0165207, -0.0177866, -0.0165612, -0.0012180, 0.0012659
4: -0.0177742, -0.0167241, -0.0177143, -0.0167369, -0.0010372, 0.0009902

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200382, -0.0196719, -0.0005594, 0.0003433
1: -0.0186527, -0.0175739, -0.0185494, -0.0176000, -0.0010528, 0.0009755
2: -0.0187115, -0.0175769, -0.0185388, -0.0175805, -0.0011310, 0.0009619
3: -0.0178239, -0.0165320, -0.0177483, -0.0165663, -0.0012576, 0.0012162
4: -0.0178280, -0.0167218, -0.0176821, -0.0167244, -0.0011036, 0.0009603

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200382, -0.0196719, -0.0004782, 0.0003655
1: -0.0186027, -0.0175448, -0.0185494, -0.0176000, -0.0010028, 0.0010046
2: -0.0186596, -0.0175877, -0.0185388, -0.0175805, -0.0010791, 0.0009511
3: -0.0177792, -0.0165207, -0.0177483, -0.0165663, -0.0012129, 0.0012276
4: -0.0177742, -0.0167241, -0.0176821, -0.0167244, -0.0010498, 0.0009580

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201467, -0.0196733, -0.0200946, -0.0196850, -0.0004618, 0.0004213
1: -0.0185973, -0.0175479, -0.0186069, -0.0176012, -0.0009961, 0.0010590
2: -0.0186375, -0.0175907, -0.0185809, -0.0175803, -0.0010572, 0.0009902
3: -0.0177729, -0.0165232, -0.0177866, -0.0165612, -0.0012117, 0.0012634
4: -0.0177583, -0.0167275, -0.0177143, -0.0167369, -0.0010213, 0.0009868

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200405, -0.0196715, -0.0005598, 0.0003456
1: -0.0186527, -0.0175739, -0.0185575, -0.0175971, -0.0010556, 0.0009836
2: -0.0187115, -0.0175769, -0.0185536, -0.0175797, -0.0011318, 0.0009767
3: -0.0178239, -0.0165320, -0.0177534, -0.0165643, -0.0012596, 0.0012214
4: -0.0178280, -0.0167218, -0.0176978, -0.0167237, -0.0011044, 0.0009760

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201467, -0.0196733, -0.0200405, -0.0196715, -0.0004752, 0.0003672
1: -0.0185973, -0.0175479, -0.0185575, -0.0175971, -0.0010002, 0.0010096
2: -0.0186375, -0.0175907, -0.0185536, -0.0175797, -0.0010578, 0.0009629
3: -0.0177729, -0.0165232, -0.0177534, -0.0165643, -0.0012086, 0.0012302
4: -0.0177583, -0.0167275, -0.0176978, -0.0167237, -0.0010346, 0.0009703

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200946, -0.0196850, -0.0005463, 0.0003997
1: -0.0186527, -0.0175739, -0.0186069, -0.0176012, -0.0010516, 0.0010330
2: -0.0187115, -0.0175769, -0.0185809, -0.0175803, -0.0011312, 0.0010041
3: -0.0178239, -0.0165320, -0.0177866, -0.0165612, -0.0012627, 0.0012545
4: -0.0178280, -0.0167218, -0.0177143, -0.0167369, -0.0010911, 0.0009925

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200946, -0.0196850, -0.0004652, 0.0004219
1: -0.0186027, -0.0175448, -0.0186069, -0.0176012, -0.0010016, 0.0010621
2: -0.0186596, -0.0175877, -0.0185809, -0.0175803, -0.0010793, 0.0009933
3: -0.0177792, -0.0165207, -0.0177866, -0.0165612, -0.0012180, 0.0012659
4: -0.0177742, -0.0167241, -0.0177143, -0.0167369, -0.0010372, 0.0009902

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202313, -0.0196949, -0.0200405, -0.0196715, -0.0005598, 0.0003456
1: -0.0186527, -0.0175739, -0.0185575, -0.0175971, -0.0010556, 0.0009836
2: -0.0187115, -0.0175769, -0.0185536, -0.0175797, -0.0011318, 0.0009767
3: -0.0178239, -0.0165320, -0.0177534, -0.0165643, -0.0012596, 0.0012214
4: -0.0178280, -0.0167218, -0.0176978, -0.0167237, -0.0011044, 0.0009760

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201502, -0.0196727, -0.0200405, -0.0196715, -0.0004787, 0.0003678
1: -0.0186027, -0.0175448, -0.0185575, -0.0175971, -0.0010057, 0.0010127
2: -0.0186596, -0.0175877, -0.0185536, -0.0175797, -0.0010799, 0.0009659
3: -0.0177792, -0.0165207, -0.0177534, -0.0165643, -0.0012149, 0.0012327
4: -0.0177742, -0.0167241, -0.0176978, -0.0167237, -0.0010505, 0.0009737

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
time: 0.22 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 1.98 seconds
NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004119, upper bound: 0.0000213
NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003865, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003929, upper bound: 0.0001473
NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0000417
NS_A2_B1_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
NS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004254, upper bound: 0.0001663
NS_A2_B1_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004393, upper bound: 0.0001702
NS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004149, upper bound: 0.0001653
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.98
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0001691

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200405, -0.0196715, -0.0005394, 0.0003001
1: -0.0186439, -0.0175444, -0.0185575, -0.0175971, -0.0010468, 0.0010132
2: -0.0187389, -0.0176913, -0.0185536, -0.0175797, -0.0011592, 0.0008623
3: -0.0178018, -0.0165161, -0.0177534, -0.0165643, -0.0012375, 0.0012373
4: -0.0179121, -0.0168245, -0.0176978, -0.0167237, -0.0011885, 0.0008733

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200405, -0.0196715, -0.0004896, 0.0003021
1: -0.0186171, -0.0175008, -0.0185575, -0.0175971, -0.0010200, 0.0010567
2: -0.0186920, -0.0176950, -0.0185536, -0.0175797, -0.0011123, 0.0008586
3: -0.0177792, -0.0164873, -0.0177534, -0.0165643, -0.0012149, 0.0012661
4: -0.0178829, -0.0168297, -0.0176978, -0.0167237, -0.0011592, 0.0008681

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200382, -0.0196719, -0.0005390, 0.0002978
1: -0.0186439, -0.0175444, -0.0185494, -0.0176000, -0.0010439, 0.0010051
2: -0.0187389, -0.0176913, -0.0185388, -0.0175805, -0.0011584, 0.0008475
3: -0.0178018, -0.0165161, -0.0177483, -0.0165663, -0.0012355, 0.0012321
4: -0.0179121, -0.0168245, -0.0176821, -0.0167244, -0.0011878, 0.0008576

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200382, -0.0196719, -0.0004892, 0.0002998
1: -0.0186171, -0.0175008, -0.0185494, -0.0176000, -0.0010171, 0.0010486
2: -0.0186920, -0.0176950, -0.0185388, -0.0175805, -0.0011114, 0.0008438
3: -0.0177792, -0.0164873, -0.0177483, -0.0165663, -0.0012129, 0.0012610
4: -0.0178829, -0.0168297, -0.0176821, -0.0167244, -0.0011585, 0.0008525

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200946, -0.0196850, -0.0005259, 0.0003542
1: -0.0186439, -0.0175444, -0.0186069, -0.0176012, -0.0010427, 0.0010626
2: -0.0187389, -0.0176913, -0.0185809, -0.0175803, -0.0011585, 0.0008896
3: -0.0178018, -0.0165161, -0.0177866, -0.0165612, -0.0012406, 0.0012705
4: -0.0179121, -0.0168245, -0.0177143, -0.0167369, -0.0011752, 0.0008898

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202109, -0.0197404, -0.0200405, -0.0196715, -0.0005394, 0.0003001
1: -0.0186439, -0.0175444, -0.0185575, -0.0175971, -0.0010468, 0.0010132
2: -0.0187389, -0.0176913, -0.0185536, -0.0175797, -0.0011592, 0.0008623
3: -0.0178018, -0.0165161, -0.0177534, -0.0165643, -0.0012375, 0.0012373
4: -0.0179121, -0.0168245, -0.0176978, -0.0167237, -0.0011885, 0.0008733

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000753
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200946, -0.0196850, -0.0004762, 0.0003562
1: -0.0186171, -0.0175008, -0.0186069, -0.0176012, -0.0010159, 0.0011061
2: -0.0186920, -0.0176950, -0.0185809, -0.0175803, -0.0011116, 0.0008860
3: -0.0177792, -0.0164873, -0.0177866, -0.0165612, -0.0012180, 0.0012993
4: -0.0178829, -0.0168297, -0.0177143, -0.0167369, -0.0011459, 0.0008846

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004117, upper bound: 0.0000202
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0197384, -0.0200405, -0.0196715, -0.0004896, 0.0003021
1: -0.0186171, -0.0175008, -0.0185575, -0.0175971, -0.0010200, 0.0010567
2: -0.0186920, -0.0176950, -0.0185536, -0.0175797, -0.0011123, 0.0008586
3: -0.0177792, -0.0164873, -0.0177534, -0.0165643, -0.0012149, 0.0012661
4: -0.0178829, -0.0168297, -0.0176978, -0.0167237, -0.0011592, 0.0008681

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004152, upper bound: 0.0000202
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200405, -0.0196715, -0.0005494, 0.0003616
1: -0.0186431, -0.0175629, -0.0185575, -0.0175971, -0.0010460, 0.0009946
2: -0.0187184, -0.0175800, -0.0185536, -0.0175797, -0.0011386, 0.0009736
3: -0.0178262, -0.0165264, -0.0177534, -0.0165643, -0.0012619, 0.0012270
4: -0.0178274, -0.0167230, -0.0176978, -0.0167237, -0.0011037, 0.0009748

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200405, -0.0196715, -0.0004755, 0.0003609
1: -0.0186148, -0.0175297, -0.0185575, -0.0175971, -0.0010177, 0.0010278
2: -0.0186554, -0.0175794, -0.0185536, -0.0175797, -0.0010757, 0.0009742
3: -0.0178018, -0.0165113, -0.0177534, -0.0165643, -0.0012375, 0.0012421
4: -0.0177744, -0.0167240, -0.0176978, -0.0167237, -0.0010508, 0.0009738

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200382, -0.0196719, -0.0005489, 0.0003593
1: -0.0186431, -0.0175629, -0.0185494, -0.0176000, -0.0010431, 0.0009865
2: -0.0187184, -0.0175800, -0.0185388, -0.0175805, -0.0011378, 0.0009588
3: -0.0178262, -0.0165264, -0.0177483, -0.0165663, -0.0012599, 0.0012219
4: -0.0178274, -0.0167230, -0.0176821, -0.0167244, -0.0011030, 0.0009592

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200382, -0.0196719, -0.0004750, 0.0003585
1: -0.0186148, -0.0175297, -0.0185494, -0.0176000, -0.0010149, 0.0010197
2: -0.0186554, -0.0175794, -0.0185388, -0.0175805, -0.0010749, 0.0009594
3: -0.0178018, -0.0165113, -0.0177483, -0.0165663, -0.0012355, 0.0012370
4: -0.0177744, -0.0167240, -0.0176821, -0.0167244, -0.0010500, 0.0009581

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200946, -0.0196850, -0.0005359, 0.0004157
1: -0.0186431, -0.0175629, -0.0186069, -0.0176012, -0.0010419, 0.0010440
2: -0.0187184, -0.0175800, -0.0185809, -0.0175803, -0.0011380, 0.0010009
3: -0.0178262, -0.0165264, -0.0177866, -0.0165612, -0.0012650, 0.0012602
4: -0.0178274, -0.0167230, -0.0177143, -0.0167369, -0.0010904, 0.0009913

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001618
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202209, -0.0196789, -0.0200405, -0.0196715, -0.0005494, 0.0003616
1: -0.0186431, -0.0175629, -0.0185575, -0.0175971, -0.0010460, 0.0009946
2: -0.0187184, -0.0175800, -0.0185536, -0.0175797, -0.0011386, 0.0009736
3: -0.0178262, -0.0165264, -0.0177534, -0.0165643, -0.0012619, 0.0012270
4: -0.0178274, -0.0167230, -0.0176978, -0.0167237, -0.0011037, 0.0009748

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001618
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200946, -0.0196850, -0.0004620, 0.0004150
1: -0.0186148, -0.0175297, -0.0186069, -0.0176012, -0.0010137, 0.0010772
2: -0.0186554, -0.0175794, -0.0185809, -0.0175803, -0.0010751, 0.0010015
3: -0.0178018, -0.0165113, -0.0177866, -0.0165612, -0.0012406, 0.0012753
4: -0.0177744, -0.0167240, -0.0177143, -0.0167369, -0.0010375, 0.0009903

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003864, upper bound: 0.0001463
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003928, upper bound: 0.0001463
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201470, -0.0196796, -0.0200405, -0.0196715, -0.0004755, 0.0003609
1: -0.0186148, -0.0175297, -0.0185575, -0.0175971, -0.0010177, 0.0010278
2: -0.0186554, -0.0175794, -0.0185536, -0.0175797, -0.0010757, 0.0009742
3: -0.0178018, -0.0165113, -0.0177534, -0.0165643, -0.0012375, 0.0012421
4: -0.0177744, -0.0167240, -0.0176978, -0.0167237, -0.0010508, 0.0009738

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 44

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003899, upper bound: 0.0001463
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003963, upper bound: 0.0001463
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200946, -0.0196850, -0.0004893, 0.0003590
1: -0.0186353, -0.0175003, -0.0186069, -0.0176012, -0.0010342, 0.0011066
2: -0.0187326, -0.0176718, -0.0185809, -0.0175803, -0.0011522, 0.0009092
3: -0.0178025, -0.0164775, -0.0177866, -0.0165612, -0.0012413, 0.0013091
4: -0.0179169, -0.0168191, -0.0177143, -0.0167369, -0.0011799, 0.0008952

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200382, -0.0196719, -0.0005023, 0.0003026
1: -0.0186353, -0.0175003, -0.0185494, -0.0176000, -0.0010354, 0.0010491
2: -0.0187326, -0.0176718, -0.0185388, -0.0175805, -0.0011520, 0.0008670
3: -0.0178025, -0.0164775, -0.0177483, -0.0165663, -0.0012362, 0.0012707
4: -0.0179169, -0.0168191, -0.0176821, -0.0167244, -0.0011925, 0.0008630

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200946, -0.0196850, -0.0004893, 0.0003590
1: -0.0186353, -0.0175003, -0.0186069, -0.0176012, -0.0010342, 0.0011066
2: -0.0187326, -0.0176718, -0.0185809, -0.0175803, -0.0011522, 0.0009092
3: -0.0178025, -0.0164775, -0.0177866, -0.0165612, -0.0012413, 0.0013091
4: -0.0179169, -0.0168191, -0.0177143, -0.0167369, -0.0011799, 0.0008952

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201742, -0.0197356, -0.0200382, -0.0196719, -0.0005023, 0.0003026
1: -0.0186353, -0.0175003, -0.0185494, -0.0176000, -0.0010354, 0.0010491
2: -0.0187326, -0.0176718, -0.0185388, -0.0175805, -0.0011520, 0.0008670
3: -0.0178025, -0.0164775, -0.0177483, -0.0165663, -0.0012362, 0.0012707
4: -0.0179169, -0.0168191, -0.0176821, -0.0167244, -0.0011925, 0.0008630

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200946, -0.0196850, -0.0004968, 0.0003610
1: -0.0186521, -0.0174887, -0.0186069, -0.0176012, -0.0010510, 0.0011183
2: -0.0188297, -0.0176612, -0.0185809, -0.0175803, -0.0012494, 0.0009197
3: -0.0178194, -0.0164689, -0.0177866, -0.0165612, -0.0012582, 0.0013177
4: -0.0180115, -0.0168087, -0.0177143, -0.0167369, -0.0012745, 0.0009056

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201818, -0.0197337, -0.0200382, -0.0196719, -0.0005098, 0.0003045
1: -0.0186521, -0.0174887, -0.0185494, -0.0176000, -0.0010522, 0.0010608
2: -0.0188297, -0.0176612, -0.0185388, -0.0175805, -0.0012492, 0.0008775
3: -0.0178194, -0.0164689, -0.0177483, -0.0165663, -0.0012532, 0.0012794
4: -0.0180115, -0.0168087, -0.0176821, -0.0167244, -0.0012871, 0.0008735

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004341, upper bound: 0.0000314
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200946, -0.0196850, -0.0005634, 0.0003503
1: -0.0186901, -0.0175541, -0.0186069, -0.0176012, -0.0010890, 0.0010528
2: -0.0188018, -0.0176594, -0.0185809, -0.0175803, -0.0012215, 0.0009216
3: -0.0178560, -0.0165166, -0.0177866, -0.0165612, -0.0012948, 0.0012700
4: -0.0179617, -0.0168128, -0.0177143, -0.0167369, -0.0012248, 0.0009015

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000526
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004136, upper bound: 0.0000314
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202484, -0.0197443, -0.0200382, -0.0196719, -0.0005765, 0.0002939
1: -0.0186901, -0.0175541, -0.0185494, -0.0176000, -0.0010902, 0.0009953
2: -0.0188018, -0.0176594, -0.0185388, -0.0175805, -0.0012213, 0.0008794
3: -0.0178560, -0.0165166, -0.0177483, -0.0165663, -0.0012897, 0.0012317
4: -0.0179617, -0.0168128, -0.0176821, -0.0167244, -0.0012373, 0.0008693

Time for backsubstitution: 1.33 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.70 + 419.58 = 421.28 seconds
