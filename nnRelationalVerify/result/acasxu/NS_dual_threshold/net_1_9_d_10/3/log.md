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
execution time: IAR + RelationalAnalysis = 1.57 + 0.55 = 2.11 seconds
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
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0002052, upper bound: 0.0004755
time: 0.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004761, upper bound: 0.0004761
time: 0.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.48 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0002052, upper bound: 0.0004755
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0004761, upper bound: 0.0004761

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0200573, -0.0196402, -0.0201701, -0.0196344, -0.0004229, 0.0005298
1: -0.0185711, -0.0175256, -0.0186188, -0.0174458, -0.0011253, 0.0010932
2: -0.0186060, -0.0175360, -0.0187211, -0.0175092, -0.0010968, 0.0011851
3: -0.0177788, -0.0165006, -0.0178066, -0.0164318, -0.0013470, 0.0013059
4: -0.0177621, -0.0166911, -0.0178369, -0.0166623, -0.0010999, 0.0011458

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 9
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30

Time for candidate selection: 0.11 seconds

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
time: 0.14 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0201701, -0.0196344, -0.0005354, 0.0005355
1: -0.0186186, -0.0174470, -0.0186188, -0.0174458, -0.0011728, 0.0011718
2: -0.0187207, -0.0175095, -0.0187211, -0.0175092, -0.0012114, 0.0012116
3: -0.0178064, -0.0164329, -0.0178066, -0.0164318, -0.0013745, 0.0013737
4: -0.0178364, -0.0166625, -0.0178369, -0.0166623, -0.0011741, 0.0011743

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 3, pos: 9

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052
time: 0.14 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052
time: 0.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.23 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0002046, upper bound: 0.0002046
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0002046, upper bound: 0.0002046
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0004755, upper bound: 0.0002052

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0200573, -0.0196402, -0.0005297, 0.0004228
1: -0.0186186, -0.0174470, -0.0185711, -0.0175256, -0.0010930, 0.0011241
2: -0.0187207, -0.0175095, -0.0186060, -0.0175360, -0.0011846, 0.0010965
3: -0.0178064, -0.0164329, -0.0177788, -0.0165006, -0.0013058, 0.0013459
4: -0.0178364, -0.0166625, -0.0177621, -0.0166911, -0.0011453, 0.0010996

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004616, upper bound: 0.0001973
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
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

### NS candidates at layer 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0002956
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0003303
time: 0.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.31 seconds
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0004616, upper bound: 0.0001973
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0002019
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0002956
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0004741, upper bound: 0.0003303

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0200512, -0.0196471, -0.0005228, 0.0004167
1: -0.0186186, -0.0174470, -0.0185725, -0.0175298, -0.0010888, 0.0011255
2: -0.0187207, -0.0175095, -0.0185882, -0.0175426, -0.0011781, 0.0010788
3: -0.0178064, -0.0164329, -0.0177999, -0.0165085, -0.0012979, 0.0013670
4: -0.0178364, -0.0166625, -0.0177291, -0.0166914, -0.0011450, 0.0010666

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004441, upper bound: 0.0001693
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 25

### Candidate
type: A, layer: 5, pos: 6

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 19

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 19

### Candidate
type: A, layer: 5, pos: 44

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

### Candidate
type: A, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0003889, upper bound: 0.0001682
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004578, upper bound: 0.0001973
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0200546, -0.0196417, -0.0005282, 0.0004201
1: -0.0186186, -0.0174470, -0.0185629, -0.0175361, -0.0010825, 0.0011159
2: -0.0187207, -0.0175095, -0.0185846, -0.0175404, -0.0011803, 0.0010751
3: -0.0178064, -0.0164329, -0.0177695, -0.0165109, -0.0012954, 0.0013366
4: -0.0178364, -0.0166625, -0.0177360, -0.0166955, -0.0011409, 0.0010735

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
time: 0.14 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0201699, -0.0196345, -0.0005275, 0.0005287
1: -0.0186205, -0.0174573, -0.0186186, -0.0174470, -0.0011735, 0.0011613
2: -0.0186928, -0.0175274, -0.0187207, -0.0175095, -0.0011833, 0.0011932
3: -0.0178194, -0.0164468, -0.0178064, -0.0164329, -0.0013865, 0.0013596
4: -0.0178173, -0.0166758, -0.0178364, -0.0166625, -0.0011547, 0.0011606

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201653, -0.0196361, -0.0201699, -0.0196345, -0.0005308, 0.0005338
1: -0.0186108, -0.0174589, -0.0186186, -0.0174470, -0.0011638, 0.0011597
2: -0.0186998, -0.0175154, -0.0187207, -0.0175095, -0.0011903, 0.0012052
3: -0.0177972, -0.0164446, -0.0178064, -0.0164329, -0.0013643, 0.0013618
4: -0.0178163, -0.0166688, -0.0178364, -0.0166625, -0.0011537, 0.0011676

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 3

No NS candidates found

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0002840
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0003268
time: 0.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.25 seconds
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0003889, upper bound: 0.0001682
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0004578, upper bound: 0.0001973
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0002840
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0003268

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0201632, -0.0196397, -0.0200512, -0.0196471, -0.0005162, 0.0004115
1: -0.0186088, -0.0174865, -0.0185725, -0.0175298, -0.0010791, 0.0010859
2: -0.0186747, -0.0175002, -0.0185882, -0.0175426, -0.0011322, 0.0010881
3: -0.0177907, -0.0164851, -0.0177999, -0.0165085, -0.0012821, 0.0013148
4: -0.0177900, -0.0166576, -0.0177291, -0.0166914, -0.0010986, 0.0010715

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

### Candidate
type: A, layer: 5, pos: 25

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: B, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 19

### Candidate
type: B, layer: 5, pos: 19

### Candidate
type: B, layer: 5, pos: 44

### Candidate
type: A, layer: 5, pos: 44

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201601, -0.0196373, -0.0200512, -0.0196471, -0.0005131, 0.0004139
1: -0.0186150, -0.0174633, -0.0185725, -0.0175298, -0.0010852, 0.0011091
2: -0.0186891, -0.0175135, -0.0185882, -0.0175426, -0.0011466, 0.0010747
3: -0.0177974, -0.0164507, -0.0177999, -0.0165085, -0.0012889, 0.0013492
4: -0.0178081, -0.0166667, -0.0177291, -0.0166914, -0.0011167, 0.0010624

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

### Candidate
type: A, layer: 5, pos: 25

### Candidate
type: A, layer: 5, pos: 6

### Candidate
type: B, layer: 5, pos: 6

### Candidate
type: A, layer: 5, pos: 19

### Candidate
type: B, layer: 5, pos: 19

### Candidate
type: B, layer: 5, pos: 44

### Candidate
type: A, layer: 5, pos: 44

### Candidate
type: B, layer: 5, pos: 30

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0200946, -0.0196850, -0.0004849, 0.0004601
1: -0.0186186, -0.0174470, -0.0186069, -0.0176012, -0.0010174, 0.0011600
2: -0.0187207, -0.0175095, -0.0185809, -0.0175803, -0.0011403, 0.0010714
3: -0.0178064, -0.0164329, -0.0177866, -0.0165612, -0.0012452, 0.0013537
4: -0.0178364, -0.0166625, -0.0177143, -0.0167369, -0.0010995, 0.0010518

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001728
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0201699, -0.0196345, -0.0200405, -0.0196715, -0.0004984, 0.0004060
1: -0.0186186, -0.0174470, -0.0185575, -0.0175971, -0.0010215, 0.0011105
2: -0.0187207, -0.0175095, -0.0185536, -0.0175797, -0.0011409, 0.0010441
3: -0.0178064, -0.0164329, -0.0177534, -0.0165643, -0.0012421, 0.0013206
4: -0.0178364, -0.0166625, -0.0176978, -0.0167237, -0.0011127, 0.0010353

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001728
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0201621, -0.0196412, -0.0201687, -0.0196408, -0.0005213, 0.0005276
1: -0.0186205, -0.0174573, -0.0186160, -0.0174732, -0.0011473, 0.0011587
2: -0.0186928, -0.0175274, -0.0187124, -0.0175415, -0.0011513, 0.0011850
3: -0.0178194, -0.0164468, -0.0178019, -0.0164559, -0.0013635, 0.0013552
4: -0.0178173, -0.0166758, -0.0178270, -0.0166876, -0.0011297, 0.0011513

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002571
time: 0.15 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002796
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0201848, -0.0197026, -0.0201699, -0.0196345, -0.0005503, 0.0004673
1: -0.0186559, -0.0174389, -0.0186186, -0.0174470, -0.0012090, 0.0011797
2: -0.0188510, -0.0176227, -0.0187207, -0.0175095, -0.0013415, 0.0010979
3: -0.0178288, -0.0164218, -0.0178064, -0.0164329, -0.0013959, 0.0013846
4: -0.0180366, -0.0167791, -0.0178364, -0.0166625, -0.0013741, 0.0010573

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0201641, -0.0196424, -0.0201699, -0.0196345, -0.0005296, 0.0005275
1: -0.0186081, -0.0174854, -0.0186186, -0.0174470, -0.0011611, 0.0011332
2: -0.0186916, -0.0175474, -0.0187207, -0.0175095, -0.0011821, 0.0011733
3: -0.0177927, -0.0164679, -0.0178064, -0.0164329, -0.0013599, 0.0013385
4: -0.0178068, -0.0166938, -0.0178364, -0.0166625, -0.0011442, 0.0011426

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268
time: 0.16 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268
time: 0.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.25 seconds
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001728
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001728
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002796
NS_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
NS_A2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0002840
NS_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268
NS_A2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 1.25
Output dim: 0, lower bound: -0.0003658, upper bound: 0.0003268

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200405, -0.0196715, -0.0004852, 0.0003756
1: -0.0186133, -0.0175046, -0.0185575, -0.0175971, -0.0010162, 0.0010529
2: -0.0186904, -0.0175496, -0.0185536, -0.0175797, -0.0011107, 0.0010040
3: -0.0177928, -0.0164827, -0.0177534, -0.0165643, -0.0012285, 0.0012707
4: -0.0178026, -0.0166927, -0.0176978, -0.0167237, -0.0010790, 0.0010051

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0201687, -0.0196408, -0.0005227, 0.0004616
1: -0.0186202, -0.0174565, -0.0186160, -0.0174732, -0.0011471, 0.0011596
2: -0.0187094, -0.0176630, -0.0187124, -0.0175415, -0.0011679, 0.0010494
3: -0.0177892, -0.0164457, -0.0178019, -0.0164559, -0.0013333, 0.0013562
4: -0.0179050, -0.0168030, -0.0178270, -0.0166876, -0.0012174, 0.0010241

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

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
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201687, -0.0196408, -0.0005204, 0.0005229
1: -0.0186195, -0.0174784, -0.0186160, -0.0174732, -0.0011463, 0.0011376
2: -0.0186864, -0.0175455, -0.0187124, -0.0175415, -0.0011450, 0.0011669
3: -0.0178169, -0.0164651, -0.0178019, -0.0164559, -0.0013610, 0.0013368
4: -0.0178100, -0.0166962, -0.0178270, -0.0166876, -0.0011224, 0.0011308

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002796
time: 0.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.60 seconds
NS_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.60
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002796

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200405, -0.0196715, -0.0004852, 0.0003756
1: -0.0186133, -0.0175046, -0.0185575, -0.0175971, -0.0010162, 0.0010529
2: -0.0186904, -0.0175496, -0.0185536, -0.0175797, -0.0011107, 0.0010040
3: -0.0177928, -0.0164827, -0.0177534, -0.0165643, -0.0012285, 0.0012707
4: -0.0178026, -0.0166927, -0.0176978, -0.0167237, -0.0010790, 0.0010051

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201687, -0.0196408, -0.0005204, 0.0005229
1: -0.0186195, -0.0174784, -0.0186160, -0.0174732, -0.0011463, 0.0011376
2: -0.0186864, -0.0175455, -0.0187124, -0.0175415, -0.0011450, 0.0011669
3: -0.0178169, -0.0164651, -0.0178019, -0.0164559, -0.0013610, 0.0013368
4: -0.0178100, -0.0166962, -0.0178270, -0.0166876, -0.0011224, 0.0011308

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002571
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795
time: 0.22 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.98 seconds
NS_A2_B1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
NS_A2_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.98
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200405, -0.0196715, -0.0004852, 0.0003756
1: -0.0186133, -0.0175046, -0.0185575, -0.0175971, -0.0010162, 0.0010529
2: -0.0186904, -0.0175496, -0.0185536, -0.0175797, -0.0011107, 0.0010040
3: -0.0177928, -0.0164827, -0.0177534, -0.0165643, -0.0012285, 0.0012707
4: -0.0178026, -0.0166927, -0.0176978, -0.0167237, -0.0010790, 0.0010051

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0201687, -0.0196408, -0.0005227, 0.0004616
1: -0.0186202, -0.0174565, -0.0186160, -0.0174732, -0.0011471, 0.0011596
2: -0.0187094, -0.0176630, -0.0187124, -0.0175415, -0.0011679, 0.0010494
3: -0.0177892, -0.0164457, -0.0178019, -0.0164559, -0.0013333, 0.0013562
4: -0.0179050, -0.0168030, -0.0178270, -0.0166876, -0.0012174, 0.0010241

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201687, -0.0196408, -0.0005204, 0.0005229
1: -0.0186195, -0.0174784, -0.0186160, -0.0174732, -0.0011463, 0.0011376
2: -0.0186864, -0.0175455, -0.0187124, -0.0175415, -0.0011450, 0.0011669
3: -0.0178169, -0.0164651, -0.0178019, -0.0164559, -0.0013610, 0.0013368
4: -0.0178100, -0.0166962, -0.0178270, -0.0166876, -0.0011224, 0.0011308

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795
time: 0.20 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.72 seconds
NS_A2_B1_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
NS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.72
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200405, -0.0196715, -0.0004852, 0.0003756
1: -0.0186133, -0.0175046, -0.0185575, -0.0175971, -0.0010162, 0.0010529
2: -0.0186904, -0.0175496, -0.0185536, -0.0175797, -0.0011107, 0.0010040
3: -0.0177928, -0.0164827, -0.0177534, -0.0165643, -0.0012285, 0.0012707
4: -0.0178026, -0.0166927, -0.0176978, -0.0167237, -0.0010790, 0.0010051

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201687, -0.0196408, -0.0005204, 0.0005229
1: -0.0186195, -0.0174784, -0.0186160, -0.0174732, -0.0011463, 0.0011376
2: -0.0186864, -0.0175455, -0.0187124, -0.0175415, -0.0011450, 0.0011669
3: -0.0178169, -0.0164651, -0.0178019, -0.0164559, -0.0013610, 0.0013368
4: -0.0178100, -0.0166962, -0.0178270, -0.0166876, -0.0011224, 0.0011308

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 5
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002571
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795
time: 0.19 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 1.49 seconds
NS_A2_B1_B2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
NS_A2_B1_B2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
NS_A2_B1_B2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
NS_A2_B1_B2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
NS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.49
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002795

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200405, -0.0196715, -0.0004852, 0.0003756
1: -0.0186133, -0.0175046, -0.0185575, -0.0175971, -0.0010162, 0.0010529
2: -0.0186904, -0.0175496, -0.0185536, -0.0175797, -0.0011107, 0.0010040
3: -0.0177928, -0.0164827, -0.0177534, -0.0165643, -0.0012285, 0.0012707
4: -0.0178026, -0.0166927, -0.0176978, -0.0167237, -0.0010790, 0.0010051

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0201635, -0.0197071, -0.0201687, -0.0196408, -0.0005227, 0.0004616
1: -0.0186202, -0.0174565, -0.0186160, -0.0174732, -0.0011471, 0.0011596
2: -0.0187094, -0.0176630, -0.0187124, -0.0175415, -0.0011679, 0.0010494
3: -0.0177892, -0.0164457, -0.0178019, -0.0164559, -0.0013333, 0.0013562
4: -0.0179050, -0.0168030, -0.0178270, -0.0166876, -0.0012174, 0.0010241

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0201611, -0.0196459, -0.0201687, -0.0196408, -0.0005204, 0.0005229
1: -0.0186195, -0.0174784, -0.0186160, -0.0174732, -0.0011463, 0.0011376
2: -0.0186864, -0.0175455, -0.0187124, -0.0175415, -0.0011450, 0.0011669
3: -0.0178169, -0.0164651, -0.0178019, -0.0164559, -0.0013610, 0.0013368
4: -0.0178100, -0.0166962, -0.0178270, -0.0166876, -0.0011224, 0.0011308

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002796
time: 0.21 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 1.61 seconds
NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004353, upper bound: 0.0001738
NS_A2_B1_B2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004487, upper bound: 0.0001739
NS_A2_B1_B2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
NS_A2_B1_B2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
NS_A2_B1_B2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004474, upper bound: 0.0001728
NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002571
NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0003194, upper bound: 0.0002795
NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.61
Output dim: 0, lower bound: -0.0004223, upper bound: 0.0002796

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.25 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.26 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.20 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.26 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.26 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200405, -0.0196715, -0.0005678, 0.0003636
1: -0.0186632, -0.0175344, -0.0185575, -0.0175971, -0.0010661, 0.0010231
2: -0.0187427, -0.0175388, -0.0185536, -0.0175797, -0.0011630, 0.0010148
3: -0.0178382, -0.0164947, -0.0177534, -0.0165643, -0.0012739, 0.0012587
4: -0.0178547, -0.0166908, -0.0176978, -0.0167237, -0.0011311, 0.0010070

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001777
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200405, -0.0196715, -0.0004807, 0.0003748
1: -0.0186059, -0.0175088, -0.0185575, -0.0175971, -0.0010089, 0.0010487
2: -0.0186617, -0.0175537, -0.0185536, -0.0175797, -0.0010820, 0.0009999
3: -0.0177845, -0.0164860, -0.0177534, -0.0165643, -0.0012202, 0.0012674
4: -0.0177835, -0.0166971, -0.0176978, -0.0167237, -0.0010598, 0.0010007

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004330, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.25 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.25 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200946, -0.0196850, -0.0004672, 0.0004289
1: -0.0186059, -0.0175088, -0.0186069, -0.0176012, -0.0010048, 0.0010982
2: -0.0186617, -0.0175537, -0.0185809, -0.0175803, -0.0010814, 0.0010273
3: -0.0177845, -0.0164860, -0.0177866, -0.0165612, -0.0012233, 0.0013006
4: -0.0177835, -0.0166971, -0.0177143, -0.0167369, -0.0010466, 0.0010172

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201521, -0.0196658, -0.0200382, -0.0196719, -0.0004802, 0.0003724
1: -0.0186059, -0.0175088, -0.0185494, -0.0176000, -0.0010060, 0.0010407
2: -0.0186617, -0.0175537, -0.0185388, -0.0175805, -0.0010812, 0.0009851
3: -0.0177845, -0.0164860, -0.0177483, -0.0165663, -0.0012182, 0.0012623
4: -0.0177835, -0.0166971, -0.0176821, -0.0167244, -0.0010591, 0.0009850

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200382, -0.0196719, -0.0004848, 0.0003732
1: -0.0186133, -0.0175046, -0.0185494, -0.0176000, -0.0010134, 0.0010448
2: -0.0186904, -0.0175496, -0.0185388, -0.0175805, -0.0011099, 0.0009892
3: -0.0177928, -0.0164827, -0.0177483, -0.0165663, -0.0012265, 0.0012656
4: -0.0178026, -0.0166927, -0.0176821, -0.0167244, -0.0010782, 0.0009895

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.22 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004466, upper bound: 0.0001728
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200946, -0.0196850, -0.0005544, 0.0004177
1: -0.0186632, -0.0175344, -0.0186069, -0.0176012, -0.0010620, 0.0010725
2: -0.0187427, -0.0175388, -0.0185809, -0.0175803, -0.0011624, 0.0010422
3: -0.0178382, -0.0164947, -0.0177866, -0.0165612, -0.0012770, 0.0012919
4: -0.0178547, -0.0166908, -0.0177143, -0.0167369, -0.0011178, 0.0010235

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 19
type: A, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202393, -0.0196769, -0.0200382, -0.0196719, -0.0005674, 0.0003613
1: -0.0186632, -0.0175344, -0.0185494, -0.0176000, -0.0010632, 0.0010150
2: -0.0187427, -0.0175388, -0.0185388, -0.0175805, -0.0011622, 0.0010000
3: -0.0178382, -0.0164947, -0.0177483, -0.0165663, -0.0012719, 0.0012536
4: -0.0178547, -0.0166908, -0.0176821, -0.0167244, -0.0011303, 0.0009913

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 5
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 19
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 19
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 46

### Candidate
type: A, layer: 5, pos: 25

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001777
time: 0.21 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004329, upper bound: 0.0001727
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0201567, -0.0196649, -0.0200946, -0.0196850, -0.0004717, 0.0004297
1: -0.0186133, -0.0175046, -0.0186069, -0.0176012, -0.0010122, 0.0011024
2: -0.0186904, -0.0175496, -0.0185809, -0.0175803, -0.0011101, 0.0010314
3: -0.0177928, -0.0164827, -0.0177866, -0.0165612, -0.0012316, 0.0013039
4: -0.0178026, -0.0166927, -0.0177143, -0.0167369, -0.0010657, 0.0010216

Time for backsubstitution: 1.20 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.11 + 418.98 = 421.09 seconds
