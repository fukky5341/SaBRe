## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00109514


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0207561, -0.0189005, -0.0207561, -0.0189005, -0.0018557, 0.0018557)
1: (-0.0197407, -0.0159238, -0.0197407, -0.0159238, -0.0038170, 0.0038170)
2: (-0.0190513, -0.0161127, -0.0190513, -0.0161127, -0.0029386, 0.0029386)
3: (-0.0188995, -0.0147928, -0.0188995, -0.0147928, -0.0041066, 0.0041066)
4: (-0.0182517, -0.0158484, -0.0182517, -0.0158484, -0.0024033, 0.0024033)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 0.60 = 1.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012884, upper bound: 0.0012884

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0012865
time: 0.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011864, upper bound: 0.0011864
time: 0.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.39 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.39
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0012865
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.39
Output dim: 0, lower bound: -0.0011864, upper bound: 0.0011864

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0207003, -0.0189312, -0.0207561, -0.0189005, -0.0017999, 0.0018250
1: -0.0195976, -0.0159658, -0.0197407, -0.0159238, -0.0036738, 0.0037750
2: -0.0189630, -0.0162635, -0.0190513, -0.0161127, -0.0028503, 0.0027878
3: -0.0187642, -0.0148378, -0.0188995, -0.0147928, -0.0039714, 0.0040617
4: -0.0181686, -0.0159423, -0.0182517, -0.0158484, -0.0023202, 0.0023094

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011863
time: 0.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011864
time: 0.17 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0208383, -0.0183058, -0.0207561, -0.0189005, -0.0019378, 0.0024504
1: -0.0204361, -0.0154199, -0.0197407, -0.0159238, -0.0045123, 0.0043209
2: -0.0192312, -0.0145860, -0.0190513, -0.0161127, -0.0031185, 0.0044652
3: -0.0192248, -0.0143090, -0.0188995, -0.0147928, -0.0044320, 0.0045905
4: -0.0184183, -0.0150544, -0.0182517, -0.0158484, -0.0025699, 0.0031974

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
time: 0.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.39 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011863
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011864
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0207003, -0.0189312, -0.0207003, -0.0189312, -0.0017691, 0.0017691
1: -0.0195976, -0.0159658, -0.0195976, -0.0159658, -0.0036318, 0.0036318
2: -0.0189630, -0.0162635, -0.0189630, -0.0162635, -0.0026995, 0.0026995
3: -0.0187642, -0.0148378, -0.0187642, -0.0148378, -0.0039264, 0.0039264
4: -0.0181686, -0.0159423, -0.0181686, -0.0159423, -0.0022263, 0.0022263

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012846
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0012426
time: 0.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0207003, -0.0189312, -0.0208383, -0.0183058, -0.0023945, 0.0019071
1: -0.0195976, -0.0159658, -0.0204361, -0.0154199, -0.0041777, 0.0044703
2: -0.0189630, -0.0162635, -0.0192312, -0.0145860, -0.0043769, 0.0029676
3: -0.0187642, -0.0148378, -0.0192248, -0.0143090, -0.0044553, 0.0043870
4: -0.0181686, -0.0159423, -0.0184183, -0.0150544, -0.0031142, 0.0024760

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010603, upper bound: 0.0012219
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010626, upper bound: 0.0012825
time: 0.16 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0208383, -0.0183058, -0.0205573, -0.0192986, -0.0015397, 0.0022515
1: -0.0204361, -0.0154199, -0.0191588, -0.0163198, -0.0041163, 0.0037389
2: -0.0192312, -0.0145860, -0.0187710, -0.0171715, -0.0020597, 0.0041849
3: -0.0192248, -0.0143090, -0.0183213, -0.0151966, -0.0040282, 0.0040123
4: -0.0184183, -0.0150544, -0.0179984, -0.0163569, -0.0020614, 0.0029440

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.00 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012846
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0012426
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0010603, upper bound: 0.0012219
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0010626, upper bound: 0.0012825
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0207003, -0.0189312, -0.0017048, 0.0016775
1: -0.0194401, -0.0160797, -0.0195976, -0.0159658, -0.0034743, 0.0035178
2: -0.0188785, -0.0165070, -0.0189630, -0.0162635, -0.0026150, 0.0024560
3: -0.0186207, -0.0149565, -0.0187642, -0.0148378, -0.0037829, 0.0038077
4: -0.0180801, -0.0160650, -0.0181686, -0.0159423, -0.0021378, 0.0021037

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012141, upper bound: 0.0012827
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012223, upper bound: 0.0012846
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0207003, -0.0189312, -0.0016941, 0.0016020
1: -0.0193564, -0.0161429, -0.0195976, -0.0159658, -0.0033906, 0.0034547
2: -0.0188713, -0.0166450, -0.0189630, -0.0162635, -0.0026078, 0.0023180
3: -0.0185336, -0.0150192, -0.0187642, -0.0148378, -0.0036958, 0.0037450
4: -0.0180798, -0.0161206, -0.0181686, -0.0159423, -0.0021375, 0.0020480

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012201, upper bound: 0.0012143
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0205079, -0.0193152, -0.0208383, -0.0183058, -0.0022021, 0.0015231
1: -0.0190408, -0.0163552, -0.0204361, -0.0154199, -0.0036209, 0.0040809
2: -0.0187032, -0.0172328, -0.0192312, -0.0145860, -0.0041172, 0.0019984
3: -0.0181952, -0.0152308, -0.0192248, -0.0143090, -0.0038862, 0.0039940
4: -0.0179204, -0.0163972, -0.0184183, -0.0150544, -0.0028661, 0.0020211

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010575, upper bound: 0.0012218
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010488, upper bound: 0.0011776
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010603, upper bound: 0.0012219
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010603, upper bound: 0.0012219
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0206639, -0.0189621, -0.0208383, -0.0183058, -0.0023581, 0.0018762
1: -0.0194627, -0.0159941, -0.0204361, -0.0154199, -0.0040429, 0.0044420
2: -0.0189031, -0.0164002, -0.0192312, -0.0145860, -0.0043170, 0.0028310
3: -0.0186207, -0.0148673, -0.0192248, -0.0143090, -0.0043118, 0.0043575
4: -0.0181086, -0.0160249, -0.0184183, -0.0150544, -0.0030543, 0.0023934

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010609, upper bound: 0.0012824
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010480, upper bound: 0.0012819
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010626, upper bound: 0.0012825
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010626, upper bound: 0.0012825
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0208383, -0.0183058, -0.0205079, -0.0193152, -0.0015231, 0.0022021
1: -0.0204361, -0.0154199, -0.0190408, -0.0163552, -0.0040809, 0.0036209
2: -0.0192312, -0.0145860, -0.0187032, -0.0172328, -0.0019984, 0.0041172
3: -0.0192248, -0.0143090, -0.0181952, -0.0152308, -0.0039940, 0.0038862
4: -0.0184183, -0.0150544, -0.0179204, -0.0163972, -0.0020211, 0.0028661

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010830, upper bound: 0.0010488
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0208383, -0.0183058, -0.0205565, -0.0188737, -0.0019645, 0.0022507
1: -0.0204361, -0.0154199, -0.0192838, -0.0158851, -0.0045510, 0.0038639
2: -0.0192312, -0.0145860, -0.0188336, -0.0161727, -0.0030585, 0.0042476
3: -0.0192248, -0.0143090, -0.0182957, -0.0147442, -0.0044806, 0.0039867
4: -0.0184183, -0.0150544, -0.0180493, -0.0158971, -0.0025212, 0.0029950

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.17 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
time: 0.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.56 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0012141, upper bound: 0.0012827
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0012223, upper bound: 0.0012846
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0012201, upper bound: 0.0012143
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0010603, upper bound: 0.0012219
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0010603, upper bound: 0.0012219
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0010626, upper bound: 0.0012825
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0010626, upper bound: 0.0012825
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.56
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0010601

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0205079, -0.0193152, -0.0013209, 0.0014851
1: -0.0194401, -0.0160797, -0.0190408, -0.0163552, -0.0030850, 0.0029611
2: -0.0188785, -0.0165070, -0.0187032, -0.0172328, -0.0016457, 0.0021963
3: -0.0186207, -0.0149565, -0.0181952, -0.0152308, -0.0033899, 0.0032387
4: -0.0180801, -0.0160650, -0.0179204, -0.0163972, -0.0016829, 0.0018555

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0206639, -0.0189621, -0.0016740, 0.0016411
1: -0.0194401, -0.0160797, -0.0194627, -0.0159941, -0.0034460, 0.0033830
2: -0.0188785, -0.0165070, -0.0189031, -0.0164002, -0.0024783, 0.0023961
3: -0.0186207, -0.0149565, -0.0186207, -0.0148673, -0.0037533, 0.0036642
4: -0.0180801, -0.0160650, -0.0181086, -0.0160249, -0.0020552, 0.0020437

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012223, upper bound: 0.0012846
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012223, upper bound: 0.0012846
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0204637, -0.0194401, -0.0207003, -0.0189312, -0.0015325, 0.0012602
1: -0.0189258, -0.0165214, -0.0195976, -0.0159658, -0.0029601, 0.0030762
2: -0.0186343, -0.0174059, -0.0189630, -0.0162635, -0.0023708, 0.0015570
3: -0.0180489, -0.0154067, -0.0187642, -0.0148378, -0.0032111, 0.0033575
4: -0.0178518, -0.0164997, -0.0181686, -0.0159423, -0.0019095, 0.0016689

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012201, upper bound: 0.0012141
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012201, upper bound: 0.0012143
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0205907, -0.0191315, -0.0207003, -0.0189312, -0.0016595, 0.0015689
1: -0.0192223, -0.0161742, -0.0195976, -0.0159658, -0.0032565, 0.0034234
2: -0.0188063, -0.0167855, -0.0189630, -0.0162635, -0.0025428, 0.0021775
3: -0.0183844, -0.0150470, -0.0187642, -0.0148378, -0.0035466, 0.0037172
4: -0.0180167, -0.0162047, -0.0181686, -0.0159423, -0.0020744, 0.0019639

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012220, upper bound: 0.0012220
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0205079, -0.0193152, -0.0205660, -0.0187726, -0.0017353, 0.0012509
1: -0.0190408, -0.0163552, -0.0193823, -0.0158637, -0.0031771, 0.0030271
2: -0.0187032, -0.0172328, -0.0188582, -0.0158765, -0.0028267, 0.0016254
3: -0.0181952, -0.0152308, -0.0183870, -0.0147328, -0.0034624, 0.0031562
4: -0.0179204, -0.0163972, -0.0180670, -0.0157083, -0.0022122, 0.0016698

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0205079, -0.0193152, -0.0207995, -0.0183434, -0.0021645, 0.0014843
1: -0.0190408, -0.0163552, -0.0202976, -0.0154510, -0.0035898, 0.0039425
2: -0.0187032, -0.0172328, -0.0191713, -0.0147189, -0.0039843, 0.0019385
3: -0.0181952, -0.0152308, -0.0190857, -0.0143400, -0.0038552, 0.0038549
4: -0.0179204, -0.0163972, -0.0183645, -0.0151376, -0.0027828, 0.0019673

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0206639, -0.0189621, -0.0205660, -0.0187726, -0.0018913, 0.0016040
1: -0.0194627, -0.0159941, -0.0193823, -0.0158637, -0.0035990, 0.0033882
2: -0.0189031, -0.0164002, -0.0188582, -0.0158765, -0.0030266, 0.0024580
3: -0.0186207, -0.0148673, -0.0183870, -0.0147328, -0.0038879, 0.0035197
4: -0.0181086, -0.0160249, -0.0180670, -0.0157083, -0.0024004, 0.0020421

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0206639, -0.0189621, -0.0207995, -0.0183434, -0.0023205, 0.0018374
1: -0.0194627, -0.0159941, -0.0202976, -0.0154510, -0.0040117, 0.0043036
2: -0.0189031, -0.0164002, -0.0191713, -0.0147189, -0.0041842, 0.0027711
3: -0.0186207, -0.0148673, -0.0190857, -0.0143400, -0.0042807, 0.0042183
4: -0.0181086, -0.0160249, -0.0183645, -0.0151376, -0.0029710, 0.0023396

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0205660, -0.0187726, -0.0205079, -0.0193152, -0.0012509, 0.0017353
1: -0.0193823, -0.0158637, -0.0190408, -0.0163552, -0.0030271, 0.0031771
2: -0.0188582, -0.0158765, -0.0187032, -0.0172328, -0.0016254, 0.0028267
3: -0.0183870, -0.0147328, -0.0181952, -0.0152308, -0.0031562, 0.0034624
4: -0.0180670, -0.0157083, -0.0179204, -0.0163972, -0.0016698, 0.0022122

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0207995, -0.0183434, -0.0205079, -0.0193152, -0.0014843, 0.0021645
1: -0.0202976, -0.0154510, -0.0190408, -0.0163552, -0.0039425, 0.0035898
2: -0.0191713, -0.0147189, -0.0187032, -0.0172328, -0.0019385, 0.0039843
3: -0.0190857, -0.0143400, -0.0181952, -0.0152308, -0.0038549, 0.0038552
4: -0.0183645, -0.0151376, -0.0179204, -0.0163972, -0.0019673, 0.0027828

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0205660, -0.0187726, -0.0205565, -0.0188737, -0.0016923, 0.0017839
1: -0.0193823, -0.0158637, -0.0192838, -0.0158851, -0.0034972, 0.0034201
2: -0.0188582, -0.0158765, -0.0188336, -0.0161727, -0.0026855, 0.0029571
3: -0.0183870, -0.0147328, -0.0182957, -0.0147442, -0.0036428, 0.0035629
4: -0.0180670, -0.0157083, -0.0180493, -0.0158971, -0.0021699, 0.0023411

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0207995, -0.0183434, -0.0205565, -0.0188737, -0.0019257, 0.0022130
1: -0.0202976, -0.0154510, -0.0192838, -0.0158851, -0.0044126, 0.0038328
2: -0.0191713, -0.0147189, -0.0188336, -0.0161727, -0.0029986, 0.0041147
3: -0.0190857, -0.0143400, -0.0182957, -0.0147442, -0.0043415, 0.0039557
4: -0.0183645, -0.0151376, -0.0180493, -0.0158971, -0.0024673, 0.0029118

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.49 + 33.83 = 35.32 seconds
