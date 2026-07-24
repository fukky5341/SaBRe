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
execution time: IAR + RelationalAnalysis = 1.01 + 0.62 = 1.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012884, upper bound: 0.0012884

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

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

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011863
time: 0.19 seconds

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

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011864, upper bound: 0.0011863
time: 0.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011864, upper bound: 0.0011864
time: 0.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.43 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.43
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011863
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.43
Output dim: 0, lower bound: -0.0011863, upper bound: 0.0011864
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.43
Output dim: 0, lower bound: -0.0011864, upper bound: 0.0011863
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.43
Output dim: 0, lower bound: -0.0011864, upper bound: 0.0011864

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0207003, -0.0189312, -0.0207003, -0.0189312, -0.0017691, 0.0017691
1: -0.0195976, -0.0159658, -0.0195976, -0.0159658, -0.0036318, 0.0036318
2: -0.0189630, -0.0162635, -0.0189630, -0.0162635, -0.0026995, 0.0026995
3: -0.0187642, -0.0148378, -0.0187642, -0.0148378, -0.0039264, 0.0039264
4: -0.0181686, -0.0159423, -0.0181686, -0.0159423, -0.0022263, 0.0022263

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

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
time: 0.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0207003, -0.0189312, -0.0208383, -0.0183058, -0.0023945, 0.0019071
1: -0.0195976, -0.0159658, -0.0204361, -0.0154199, -0.0041777, 0.0044703
2: -0.0189630, -0.0162635, -0.0192312, -0.0145860, -0.0043769, 0.0029676
3: -0.0187642, -0.0148378, -0.0192248, -0.0143090, -0.0044553, 0.0043870
4: -0.0181686, -0.0159423, -0.0184183, -0.0150544, -0.0031142, 0.0024760

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012846
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0012426
time: 0.18 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0208383, -0.0183058, -0.0207003, -0.0189312, -0.0019071, 0.0023945
1: -0.0204361, -0.0154199, -0.0195976, -0.0159658, -0.0044703, 0.0041777
2: -0.0192312, -0.0145860, -0.0189630, -0.0162635, -0.0029676, 0.0043769
3: -0.0192248, -0.0143090, -0.0187642, -0.0148378, -0.0043870, 0.0044553
4: -0.0184183, -0.0150544, -0.0181686, -0.0159423, -0.0024760, 0.0031142

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010601, upper bound: 0.0011834
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
time: 0.16 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0208383, -0.0183058, -0.0208383, -0.0183058, -0.0025325, 0.0025325
1: -0.0204361, -0.0154199, -0.0204361, -0.0154199, -0.0050162, 0.0050162
2: -0.0192312, -0.0145860, -0.0192312, -0.0145860, -0.0046451, 0.0046451
3: -0.0192248, -0.0143090, -0.0192248, -0.0143090, -0.0049159, 0.0049159
4: -0.0184183, -0.0150544, -0.0184183, -0.0150544, -0.0033639, 0.0033639

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010601, upper bound: 0.0011834
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
time: 0.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.85 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012846
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0012426
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012846
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0011813, upper bound: 0.0012426
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0010601, upper bound: 0.0011834
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0010601, upper bound: 0.0011834
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.85
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0207003, -0.0189312, -0.0017048, 0.0016775
1: -0.0194401, -0.0160797, -0.0195976, -0.0159658, -0.0034743, 0.0035178
2: -0.0188785, -0.0165070, -0.0189630, -0.0162635, -0.0026150, 0.0024560
3: -0.0186207, -0.0149565, -0.0187642, -0.0148378, -0.0037829, 0.0038077
4: -0.0180801, -0.0160650, -0.0181686, -0.0159423, -0.0021378, 0.0021037

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0207003, -0.0189312, -0.0016941, 0.0016020
1: -0.0193564, -0.0161429, -0.0195976, -0.0159658, -0.0033906, 0.0034547
2: -0.0188713, -0.0166450, -0.0189630, -0.0162635, -0.0026078, 0.0023180
3: -0.0185336, -0.0150192, -0.0187642, -0.0148378, -0.0036958, 0.0037450
4: -0.0180798, -0.0161206, -0.0181686, -0.0159423, -0.0021375, 0.0020480

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0208383, -0.0183058, -0.0023302, 0.0018154
1: -0.0194401, -0.0160797, -0.0204361, -0.0154199, -0.0040202, 0.0043564
2: -0.0188785, -0.0165070, -0.0192312, -0.0145860, -0.0042924, 0.0027242
3: -0.0186207, -0.0149565, -0.0192248, -0.0143090, -0.0043117, 0.0042683
4: -0.0180801, -0.0160650, -0.0184183, -0.0150544, -0.0030257, 0.0023533

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011769, upper bound: 0.0012825
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010492, upper bound: 0.0012811
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0208383, -0.0183058, -0.0023195, 0.0017400
1: -0.0193564, -0.0161429, -0.0204361, -0.0154199, -0.0039365, 0.0042932
2: -0.0188713, -0.0166450, -0.0192312, -0.0145860, -0.0042852, 0.0025861
3: -0.0185336, -0.0150192, -0.0192248, -0.0143090, -0.0042246, 0.0042056
4: -0.0180798, -0.0161206, -0.0184183, -0.0150544, -0.0030254, 0.0022977

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012199
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010489, upper bound: 0.0012185
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0205660, -0.0187726, -0.0207003, -0.0189312, -0.0016348, 0.0019277
1: -0.0193823, -0.0158637, -0.0195976, -0.0159658, -0.0034165, 0.0037338
2: -0.0188582, -0.0158765, -0.0189630, -0.0162635, -0.0025947, 0.0030865
3: -0.0183870, -0.0147328, -0.0187642, -0.0148378, -0.0035492, 0.0040314
4: -0.0180670, -0.0157083, -0.0181686, -0.0159423, -0.0021247, 0.0024604

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012631, upper bound: 0.0010835
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0010603
time: 0.18 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0010626
time: 0.18 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0205660, -0.0187726, -0.0208383, -0.0183058, -0.0022602, 0.0020657
1: -0.0193823, -0.0158637, -0.0204361, -0.0154199, -0.0039624, 0.0045723
2: -0.0188582, -0.0158765, -0.0192312, -0.0145860, -0.0042721, 0.0033547
3: -0.0183870, -0.0147328, -0.0192248, -0.0143090, -0.0040781, 0.0044920
4: -0.0180670, -0.0157083, -0.0184183, -0.0150544, -0.0030127, 0.0027100

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
time: 0.18 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
time: 0.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.43 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0012444, upper bound: 0.0012444
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0011769, upper bound: 0.0012825
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0010492, upper bound: 0.0012811
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0011786, upper bound: 0.0012199
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0010489, upper bound: 0.0012185
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0010603
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0012219, upper bound: 0.0010626
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.43
Output dim: 0, lower bound: -0.0010588, upper bound: 0.0010588

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0206360, -0.0190228, -0.0016132, 0.0016132
1: -0.0194401, -0.0160797, -0.0194401, -0.0160797, -0.0033604, 0.0033604
2: -0.0188785, -0.0165070, -0.0188785, -0.0165070, -0.0023715, 0.0023715
3: -0.0186207, -0.0149565, -0.0186207, -0.0149565, -0.0036642, 0.0036642
4: -0.0180801, -0.0160650, -0.0180801, -0.0160650, -0.0020151, 0.0020151

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012438, upper bound: 0.0011632
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012429, upper bound: 0.0012838
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0206253, -0.0190983, -0.0015377, 0.0016025
1: -0.0194401, -0.0160797, -0.0193564, -0.0161429, -0.0032972, 0.0032766
2: -0.0188785, -0.0165070, -0.0188713, -0.0166450, -0.0022335, 0.0023643
3: -0.0186207, -0.0149565, -0.0185336, -0.0150192, -0.0036015, 0.0035770
4: -0.0180801, -0.0160650, -0.0180798, -0.0161206, -0.0019595, 0.0020149

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012438, upper bound: 0.0011632
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012429, upper bound: 0.0012838
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0206360, -0.0190228, -0.0016025, 0.0015377
1: -0.0193564, -0.0161429, -0.0194401, -0.0160797, -0.0032766, 0.0032972
2: -0.0188713, -0.0166450, -0.0188785, -0.0165070, -0.0023643, 0.0022335
3: -0.0185336, -0.0150192, -0.0186207, -0.0149565, -0.0035770, 0.0036015
4: -0.0180798, -0.0161206, -0.0180801, -0.0160650, -0.0020149, 0.0019595

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012436, upper bound: 0.0011873
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012418, upper bound: 0.0012418
time: 0.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0206253, -0.0190983, -0.0015270, 0.0015270
1: -0.0193564, -0.0161429, -0.0193564, -0.0161429, -0.0032134, 0.0032134
2: -0.0188713, -0.0166450, -0.0188713, -0.0166450, -0.0022262, 0.0022262
3: -0.0185336, -0.0150192, -0.0185336, -0.0150192, -0.0035144, 0.0035144
4: -0.0180798, -0.0161206, -0.0180798, -0.0161206, -0.0019592, 0.0019592

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012436, upper bound: 0.0011880
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012418, upper bound: 0.0012418
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0205660, -0.0187726, -0.0018634, 0.0015432
1: -0.0194401, -0.0160797, -0.0193823, -0.0158637, -0.0035764, 0.0033025
2: -0.0188785, -0.0165070, -0.0188582, -0.0158765, -0.0030020, 0.0023512
3: -0.0186207, -0.0149565, -0.0183870, -0.0147328, -0.0038879, 0.0034305
4: -0.0180801, -0.0160650, -0.0180670, -0.0157083, -0.0023718, 0.0020021

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010757, upper bound: 0.0012567
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0206360, -0.0190228, -0.0207995, -0.0183434, -0.0022926, 0.0017766
1: -0.0194401, -0.0160797, -0.0202976, -0.0154510, -0.0039891, 0.0042179
2: -0.0188785, -0.0165070, -0.0191713, -0.0147189, -0.0041596, 0.0026643
3: -0.0186207, -0.0149565, -0.0190857, -0.0143400, -0.0042807, 0.0041292
4: -0.0180801, -0.0160650, -0.0183645, -0.0151376, -0.0029425, 0.0022995

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0205660, -0.0187726, -0.0018527, 0.0014677
1: -0.0193564, -0.0161429, -0.0193823, -0.0158637, -0.0034926, 0.0032393
2: -0.0188713, -0.0166450, -0.0188582, -0.0158765, -0.0029948, 0.0022132
3: -0.0185336, -0.0150192, -0.0183870, -0.0147328, -0.0038008, 0.0033679
4: -0.0180798, -0.0161206, -0.0180670, -0.0157083, -0.0023716, 0.0019465

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010775, upper bound: 0.0012107
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010470, upper bound: 0.0012112
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010470, upper bound: 0.0012185
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0206253, -0.0190983, -0.0207995, -0.0183434, -0.0022819, 0.0017011
1: -0.0193564, -0.0161429, -0.0202976, -0.0154510, -0.0039053, 0.0041547
2: -0.0188713, -0.0166450, -0.0191713, -0.0147189, -0.0041524, 0.0025263
3: -0.0185336, -0.0150192, -0.0190857, -0.0143400, -0.0041936, 0.0040665
4: -0.0180798, -0.0161206, -0.0183645, -0.0151376, -0.0029422, 0.0022439

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010470, upper bound: 0.0012112
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010470, upper bound: 0.0012185
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0205660, -0.0187726, -0.0205079, -0.0193152, -0.0012509, 0.0017353
1: -0.0193823, -0.0158637, -0.0190408, -0.0163552, -0.0030271, 0.0031771
2: -0.0188582, -0.0158765, -0.0187032, -0.0172328, -0.0016254, 0.0028267
3: -0.0183870, -0.0147328, -0.0181952, -0.0152308, -0.0031562, 0.0034624
4: -0.0180670, -0.0157083, -0.0179204, -0.0163972, -0.0016698, 0.0022122

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0205660, -0.0187726, -0.0206639, -0.0189621, -0.0016040, 0.0018913
1: -0.0193823, -0.0158637, -0.0194627, -0.0159941, -0.0033882, 0.0035990
2: -0.0188582, -0.0158765, -0.0189031, -0.0164002, -0.0024580, 0.0030266
3: -0.0183870, -0.0147328, -0.0186207, -0.0148673, -0.0035197, 0.0038879
4: -0.0180670, -0.0157083, -0.0181086, -0.0160249, -0.0020421, 0.0024004

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.62 + 35.54 = 37.16 seconds
