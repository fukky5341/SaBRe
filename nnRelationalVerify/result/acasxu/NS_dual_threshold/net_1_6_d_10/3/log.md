## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.012884755


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142281, 0.0014693, -0.0142281, 0.0014693, -0.0156974, 0.0156974)
1: (-0.0323874, 0.0100724, -0.0323874, 0.0100724, -0.0424598, 0.0424598)
2: (-0.0295564, 0.0169883, -0.0295564, 0.0169883, -0.0465447, 0.0465447)
3: (-0.0313897, 0.0155250, -0.0313897, 0.0155250, -0.0469147, 0.0469147)
4: (-0.0253857, 0.0193347, -0.0253857, 0.0193347, -0.0447204, 0.0447204)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.70 + 0.75 = 1.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0135629, upper bound: 0.0135629

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133206, upper bound: 0.0132554
time: 0.18 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0133206, upper bound: 0.0132554
time: 0.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.44 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.44
Output dim: 0, lower bound: -0.0133206, upper bound: 0.0132554
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.44
Output dim: 0, lower bound: -0.0133206, upper bound: 0.0132554

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0142170, 0.0015349, -0.0142281, 0.0014693, -0.0156863, 0.0157630
1: -0.0326197, 0.0102178, -0.0323874, 0.0100724, -0.0426921, 0.0426052
2: -0.0294728, 0.0175179, -0.0295564, 0.0169883, -0.0464611, 0.0470742
3: -0.0314016, 0.0157006, -0.0313897, 0.0155250, -0.0469266, 0.0470903
4: -0.0254409, 0.0198529, -0.0253857, 0.0193347, -0.0447757, 0.0452386

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130170, upper bound: 0.0130170
time: 0.19 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130170, upper bound: 0.0130170
time: 0.20 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0140014, 0.0013007, -0.0142281, 0.0014693, -0.0154707, 0.0155288
1: -0.0318042, 0.0096758, -0.0323874, 0.0100724, -0.0418766, 0.0420632
2: -0.0291450, 0.0161654, -0.0295564, 0.0169883, -0.0461332, 0.0457218
3: -0.0308693, 0.0150180, -0.0313897, 0.0155250, -0.0463943, 0.0464077
4: -0.0249824, 0.0184374, -0.0253857, 0.0193347, -0.0443171, 0.0438231

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132554, upper bound: 0.0133206
time: 0.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132554, upper bound: 0.0133206
time: 0.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.14 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.14
Output dim: 0, lower bound: -0.0130170, upper bound: 0.0130170
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.14
Output dim: 0, lower bound: -0.0130170, upper bound: 0.0130170
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.14
Output dim: 0, lower bound: -0.0132554, upper bound: 0.0133206
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.14
Output dim: 0, lower bound: -0.0132554, upper bound: 0.0133206

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0142170, 0.0015349, -0.0142170, 0.0015349, -0.0157519, 0.0157519
1: -0.0326197, 0.0102178, -0.0326197, 0.0102178, -0.0428375, 0.0428375
2: -0.0294728, 0.0175179, -0.0294728, 0.0175179, -0.0469907, 0.0469907
3: -0.0314016, 0.0157006, -0.0314016, 0.0157006, -0.0471022, 0.0471022
4: -0.0254409, 0.0198529, -0.0254409, 0.0198529, -0.0452938, 0.0452938

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129725, upper bound: 0.0129139
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128939
time: 0.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0142170, 0.0015349, -0.0140014, 0.0013007, -0.0155176, 0.0155363
1: -0.0326197, 0.0102178, -0.0318042, 0.0096758, -0.0422955, 0.0420220
2: -0.0294728, 0.0175179, -0.0291450, 0.0161654, -0.0456383, 0.0466628
3: -0.0314016, 0.0157006, -0.0308693, 0.0150180, -0.0464196, 0.0465699
4: -0.0254409, 0.0198529, -0.0249824, 0.0184374, -0.0438784, 0.0448352

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129139, upper bound: 0.0129725
time: 0.19 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0129018
time: 0.20 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0140014, 0.0013007, -0.0142170, 0.0015349, -0.0155363, 0.0155176
1: -0.0318042, 0.0096758, -0.0326197, 0.0102178, -0.0420220, 0.0422955
2: -0.0291450, 0.0161654, -0.0294728, 0.0175179, -0.0466628, 0.0456383
3: -0.0308693, 0.0150180, -0.0314016, 0.0157006, -0.0465699, 0.0464196
4: -0.0249824, 0.0184374, -0.0254409, 0.0198529, -0.0448352, 0.0438784

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131695, upper bound: 0.0131606
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129018, upper bound: 0.0128946
time: 0.20 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0140014, 0.0013007, -0.0140014, 0.0013007, -0.0153020, 0.0153020
1: -0.0318042, 0.0096758, -0.0318042, 0.0096758, -0.0414800, 0.0414800
2: -0.0291450, 0.0161654, -0.0291450, 0.0161654, -0.0453104, 0.0453104
3: -0.0308693, 0.0150180, -0.0308693, 0.0150180, -0.0458873, 0.0458873
4: -0.0249824, 0.0184374, -0.0249824, 0.0184374, -0.0434198, 0.0434198

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132086, upper bound: 0.0115304
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0130014, upper bound: 0.0130014
time: 0.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.14 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0129725, upper bound: 0.0129139
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128939
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0129139, upper bound: 0.0129725
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0129018
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0131695, upper bound: 0.0131606
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0129018, upper bound: 0.0128946
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0132086, upper bound: 0.0115304
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.14
Output dim: 0, lower bound: -0.0130014, upper bound: 0.0130014

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0139336, 0.0011706, -0.0142170, 0.0015349, -0.0154685, 0.0153876
1: -0.0316451, 0.0095233, -0.0326197, 0.0102178, -0.0418628, 0.0421431
2: -0.0289392, 0.0161539, -0.0294728, 0.0175179, -0.0464571, 0.0456268
3: -0.0306165, 0.0148374, -0.0314016, 0.0157006, -0.0463171, 0.0462390
4: -0.0248552, 0.0183405, -0.0254409, 0.0198529, -0.0447081, 0.0437814

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128939
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128939
time: 0.19 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0142170, 0.0015349, -0.0169924, 0.0185801
1: -0.0392320, 0.0138485, -0.0326197, 0.0102178, -0.0494498, 0.0464682
2: -0.0302696, 0.0302965, -0.0294728, 0.0175179, -0.0477875, 0.0597693
3: -0.0358154, 0.0216712, -0.0314016, 0.0157006, -0.0515160, 0.0530728
4: -0.0284588, 0.0328847, -0.0254409, 0.0198529, -0.0483117, 0.0583256

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124775, upper bound: 0.0115521
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128932, upper bound: 0.0128932
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0142170, 0.0015349, -0.0137383, 0.0009519, -0.0151689, 0.0152732
1: -0.0326197, 0.0102178, -0.0309329, 0.0090424, -0.0416621, 0.0411507
2: -0.0294728, 0.0175179, -0.0286555, 0.0149123, -0.0443851, 0.0461734
3: -0.0314016, 0.0157006, -0.0301543, 0.0142417, -0.0456433, 0.0458549
4: -0.0254409, 0.0198529, -0.0244520, 0.0170415, -0.0424825, 0.0443049

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128946, upper bound: 0.0129018
time: 0.19 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128946, upper bound: 0.0129018
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0142170, 0.0015349, -0.0145102, 0.0032983, -0.0175153, 0.0160451
1: -0.0326197, 0.0102178, -0.0354606, 0.0111622, -0.0437819, 0.0456784
2: -0.0294728, 0.0175179, -0.0288524, 0.0250942, -0.0545670, 0.0463702
3: -0.0314016, 0.0157006, -0.0331143, 0.0183202, -0.0497217, 0.0488149
4: -0.0254409, 0.0198529, -0.0265385, 0.0274025, -0.0528434, 0.0463914

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128946, upper bound: 0.0129018
time: 0.18 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0129018
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0137383, 0.0009519, -0.0142170, 0.0015349, -0.0152732, 0.0151689
1: -0.0309329, 0.0090424, -0.0326197, 0.0102178, -0.0411507, 0.0416621
2: -0.0286555, 0.0149123, -0.0294728, 0.0175179, -0.0461734, 0.0443851
3: -0.0301543, 0.0142417, -0.0314016, 0.0157006, -0.0458549, 0.0456433
4: -0.0244520, 0.0170415, -0.0254409, 0.0198529, -0.0443049, 0.0424825

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145102, 0.0032983, -0.0142170, 0.0015349, -0.0160451, 0.0175153
1: -0.0354606, 0.0111622, -0.0326197, 0.0102178, -0.0456784, 0.0437819
2: -0.0288524, 0.0250942, -0.0294728, 0.0175179, -0.0463702, 0.0545670
3: -0.0331143, 0.0183202, -0.0314016, 0.0157006, -0.0488149, 0.0497217
4: -0.0265385, 0.0274025, -0.0254409, 0.0198529, -0.0463914, 0.0528434

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0118328, -0.0035663, -0.0140014, 0.0013007, -0.0131335, 0.0104351
1: -0.0206060, 0.0004676, -0.0318042, 0.0096758, -0.0302818, 0.0322718
2: -0.0244539, -0.0027796, -0.0291450, 0.0161654, -0.0406193, 0.0263653
3: -0.0195738, 0.0029438, -0.0308693, 0.0150180, -0.0345918, 0.0338131
4: -0.0203478, -0.0020269, -0.0249824, 0.0184374, -0.0387852, 0.0229555

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131687, upper bound: 0.0115005
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0132233, upper bound: 0.0115647
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0139358, 0.0012009, -0.0140014, 0.0013007, -0.0152365, 0.0152023
1: -0.0315487, 0.0094707, -0.0318042, 0.0096758, -0.0412246, 0.0412750
2: -0.0290280, 0.0158852, -0.0291450, 0.0161654, -0.0451934, 0.0450302
3: -0.0306428, 0.0146923, -0.0308693, 0.0150180, -0.0456608, 0.0455616
4: -0.0248478, 0.0181045, -0.0249824, 0.0184374, -0.0432852, 0.0430868

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117697, upper bound: 0.0135132
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117697, upper bound: 0.0135348
time: 0.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.16 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128939
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128939
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0124775, upper bound: 0.0115521
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128932, upper bound: 0.0128932
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128946, upper bound: 0.0129018
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128946, upper bound: 0.0129018
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128946, upper bound: 0.0129018
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0129018
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0128939, upper bound: 0.0128946
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0131687, upper bound: 0.0115005
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0132233, upper bound: 0.0115647
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0117697, upper bound: 0.0135132
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.16
Output dim: 0, lower bound: -0.0117697, upper bound: 0.0135348

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0139336, 0.0011706, -0.0139336, 0.0011706, -0.0151042, 0.0151042
1: -0.0316451, 0.0095233, -0.0316451, 0.0095233, -0.0411684, 0.0411684
2: -0.0289392, 0.0161539, -0.0289392, 0.0161539, -0.0450931, 0.0450931
3: -0.0306165, 0.0148374, -0.0306165, 0.0148374, -0.0454539, 0.0454539
4: -0.0248552, 0.0183405, -0.0248552, 0.0183405, -0.0431957, 0.0431957

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0129139
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0139336, 0.0011706, -0.0154575, 0.0043632, -0.0182968, 0.0166281
1: -0.0316451, 0.0095233, -0.0392320, 0.0138485, -0.0454935, 0.0487553
2: -0.0289392, 0.0161539, -0.0302696, 0.0302965, -0.0592357, 0.0464235
3: -0.0306165, 0.0148374, -0.0358154, 0.0216712, -0.0522877, 0.0506528
4: -0.0248552, 0.0183405, -0.0284588, 0.0328847, -0.0577399, 0.0467993

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0129139
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
time: 0.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0142112, 0.0015304, -0.0169879, 0.0185744
1: -0.0392320, 0.0138485, -0.0326043, 0.0101950, -0.0494270, 0.0464528
2: -0.0302696, 0.0302965, -0.0294621, 0.0174987, -0.0477683, 0.0597585
3: -0.0358154, 0.0216712, -0.0313869, 0.0156810, -0.0514964, 0.0530581
4: -0.0284588, 0.0328847, -0.0254296, 0.0198324, -0.0482912, 0.0583143

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128932, upper bound: 0.0128932
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128932, upper bound: 0.0128932
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0139336, 0.0011706, -0.0137383, 0.0009519, -0.0148855, 0.0149089
1: -0.0316451, 0.0095233, -0.0309329, 0.0090424, -0.0406875, 0.0404563
2: -0.0289392, 0.0161539, -0.0286555, 0.0149123, -0.0438515, 0.0448095
3: -0.0306165, 0.0148374, -0.0301543, 0.0142417, -0.0448582, 0.0449917
4: -0.0248552, 0.0183405, -0.0244520, 0.0170415, -0.0418968, 0.0427925

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0131570
time: 0.20 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131380, upper bound: 0.0129630
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0137383, 0.0009519, -0.0164094, 0.0181014
1: -0.0392320, 0.0138485, -0.0309329, 0.0090424, -0.0482744, 0.0447814
2: -0.0302696, 0.0302965, -0.0286555, 0.0149123, -0.0451819, 0.0589520
3: -0.0358154, 0.0216712, -0.0301543, 0.0142417, -0.0500571, 0.0518255
4: -0.0284588, 0.0328847, -0.0244520, 0.0170415, -0.0455004, 0.0573367

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131374, upper bound: 0.0122665
time: 0.19 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0131380, upper bound: 0.0129630
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0139336, 0.0011706, -0.0145102, 0.0032983, -0.0172319, 0.0156808
1: -0.0316451, 0.0095233, -0.0354606, 0.0111622, -0.0428072, 0.0449840
2: -0.0289392, 0.0161539, -0.0288524, 0.0250942, -0.0540334, 0.0450063
3: -0.0306165, 0.0148374, -0.0331143, 0.0183202, -0.0489367, 0.0479517
4: -0.0248552, 0.0183405, -0.0265385, 0.0274025, -0.0522577, 0.0448790

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0128870
time: 0.19 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127853, upper bound: 0.0127052
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0145102, 0.0032983, -0.0187558, 0.0188734
1: -0.0392320, 0.0138485, -0.0354606, 0.0111622, -0.0503942, 0.0493091
2: -0.0302696, 0.0302965, -0.0288524, 0.0250942, -0.0553638, 0.0591489
3: -0.0358154, 0.0216712, -0.0331143, 0.0183202, -0.0541356, 0.0547855
4: -0.0284588, 0.0328847, -0.0265385, 0.0274025, -0.0558613, 0.0594232

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0128870
time: 0.20 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127853, upper bound: 0.0127289
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0137383, 0.0009519, -0.0139336, 0.0011706, -0.0149089, 0.0148855
1: -0.0309329, 0.0090424, -0.0316451, 0.0095233, -0.0404563, 0.0406875
2: -0.0286555, 0.0149123, -0.0289392, 0.0161539, -0.0448095, 0.0438515
3: -0.0301543, 0.0142417, -0.0306165, 0.0148374, -0.0449917, 0.0448582
4: -0.0244520, 0.0170415, -0.0248552, 0.0183405, -0.0427925, 0.0418968

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129578, upper bound: 0.0124161
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0137383, 0.0009519, -0.0154575, 0.0043632, -0.0181014, 0.0164094
1: -0.0309329, 0.0090424, -0.0392320, 0.0138485, -0.0447814, 0.0482744
2: -0.0286555, 0.0149123, -0.0302696, 0.0302965, -0.0589520, 0.0451819
3: -0.0301543, 0.0142417, -0.0358154, 0.0216712, -0.0518255, 0.0500571
4: -0.0244520, 0.0170415, -0.0284588, 0.0328847, -0.0573367, 0.0455004

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0131374
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0129630, upper bound: 0.0131380
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145102, 0.0032983, -0.0139336, 0.0011706, -0.0156808, 0.0172319
1: -0.0354606, 0.0111622, -0.0316451, 0.0095233, -0.0449840, 0.0428072
2: -0.0288524, 0.0250942, -0.0289392, 0.0161539, -0.0450063, 0.0540334
3: -0.0331143, 0.0183202, -0.0306165, 0.0148374, -0.0479517, 0.0489367
4: -0.0265385, 0.0274025, -0.0248552, 0.0183405, -0.0448790, 0.0522577

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128792, upper bound: 0.0125041
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127052, upper bound: 0.0127853
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0145102, 0.0032983, -0.0154575, 0.0043632, -0.0188734, 0.0187558
1: -0.0354606, 0.0111622, -0.0392320, 0.0138485, -0.0493091, 0.0503942
2: -0.0288524, 0.0250942, -0.0302696, 0.0302965, -0.0591489, 0.0553638
3: -0.0331143, 0.0183202, -0.0358154, 0.0216712, -0.0547855, 0.0541356
4: -0.0265385, 0.0274025, -0.0284588, 0.0328847, -0.0594232, 0.0558613

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128870, upper bound: 0.0125143
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126973, upper bound: 0.0127853
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0118328, -0.0035663, -0.0134188, 0.0005261, -0.0123589, 0.0098526
1: -0.0206060, 0.0004676, -0.0313622, 0.0097410, -0.0303469, 0.0318298
2: -0.0244539, -0.0027796, -0.0293808, 0.0043522, -0.0288060, 0.0266012
3: -0.0195738, 0.0029438, -0.0309284, 0.0147349, -0.0343087, 0.0338722
4: -0.0203478, -0.0020269, -0.0234892, 0.0066656, -0.0270134, 0.0214623

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0116274, upper bound: 0.0115005
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0116274, upper bound: 0.0115005
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0118328, -0.0035663, -0.0137603, 0.0010983, -0.0129311, 0.0101941
1: -0.0206060, 0.0004676, -0.0313006, 0.0091799, -0.0297858, 0.0317682
2: -0.0244539, -0.0027796, -0.0287048, 0.0158141, -0.0402680, 0.0259252
3: -0.0195738, 0.0029438, -0.0303533, 0.0143960, -0.0339698, 0.0332971
4: -0.0203478, -0.0020269, -0.0245954, 0.0179614, -0.0383092, 0.0225686

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0116372, upper bound: 0.0115639
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0116372, upper bound: 0.0115647
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0139358, 0.0012009, -0.0118328, -0.0035663, -0.0103696, 0.0130337
1: -0.0315487, 0.0094707, -0.0206060, 0.0004676, -0.0320163, 0.0300767
2: -0.0290280, 0.0158852, -0.0244539, -0.0027796, -0.0262484, 0.0403391
3: -0.0306428, 0.0146923, -0.0195738, 0.0029438, -0.0335866, 0.0342661
4: -0.0248478, 0.0181045, -0.0203478, -0.0020269, -0.0228209, 0.0384523

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117664, upper bound: 0.0133411
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117571, upper bound: 0.0133397
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0139358, 0.0012009, -0.0139358, 0.0012009, -0.0151368, 0.0151368
1: -0.0315487, 0.0094707, -0.0315487, 0.0094707, -0.0410195, 0.0410195
2: -0.0290280, 0.0158852, -0.0290280, 0.0158852, -0.0449132, 0.0449132
3: -0.0306428, 0.0146923, -0.0306428, 0.0146923, -0.0453351, 0.0453351
4: -0.0248478, 0.0181045, -0.0248478, 0.0181045, -0.0429523, 0.0429523

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117664, upper bound: 0.0134469
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117571, upper bound: 0.0134474
time: 0.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.22 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0129139
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0129139
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0128932, upper bound: 0.0128932
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0128932, upper bound: 0.0128932
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0131570
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0131380, upper bound: 0.0129630
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0131374, upper bound: 0.0122665
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0131380, upper bound: 0.0129630
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0128870
NS_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0127853, upper bound: 0.0127052
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0128870
NS_A1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0127853, upper bound: 0.0127289
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0129578, upper bound: 0.0124161
NS_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0131374
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0129630, upper bound: 0.0131380
NS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0128792, upper bound: 0.0125041
NS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0127052, upper bound: 0.0127853
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0128870, upper bound: 0.0125143
NS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0126973, upper bound: 0.0127853
NS_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0116274, upper bound: 0.0115005
NS_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0116274, upper bound: 0.0115005
NS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0116372, upper bound: 0.0115639
NS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0116372, upper bound: 0.0115647
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0117664, upper bound: 0.0133411
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0117571, upper bound: 0.0133397
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0117664, upper bound: 0.0134469
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0117571, upper bound: 0.0134474

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0132636, 0.0007806, -0.0139336, 0.0011706, -0.0144342, 0.0147142
1: -0.0298148, 0.0070447, -0.0316451, 0.0095233, -0.0393381, 0.0386897
2: -0.0271393, 0.0121166, -0.0289392, 0.0161539, -0.0432932, 0.0410558
3: -0.0289018, 0.0123366, -0.0306165, 0.0148374, -0.0437392, 0.0429531
4: -0.0233742, 0.0143226, -0.0248552, 0.0183405, -0.0417147, 0.0391778

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0124161
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0127759
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0132636, 0.0007806, -0.0154575, 0.0043632, -0.0176268, 0.0162381
1: -0.0298148, 0.0070447, -0.0392320, 0.0138485, -0.0436632, 0.0462767
2: -0.0271393, 0.0121166, -0.0302696, 0.0302965, -0.0574358, 0.0423862
3: -0.0289018, 0.0123366, -0.0358154, 0.0216712, -0.0505730, 0.0481520
4: -0.0233742, 0.0143226, -0.0284588, 0.0328847, -0.0562589, 0.0427814

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0125853
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0126973
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0139283, 0.0011669, -0.0166244, 0.0182914
1: -0.0392320, 0.0138485, -0.0316318, 0.0095030, -0.0487350, 0.0454802
2: -0.0302696, 0.0302965, -0.0289299, 0.0161394, -0.0464090, 0.0592264
3: -0.0358154, 0.0216712, -0.0306028, 0.0148200, -0.0506354, 0.0522740
4: -0.0284588, 0.0328847, -0.0248460, 0.0183248, -0.0467836, 0.0577307

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128784, upper bound: 0.0124398
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126965, upper bound: 0.0127202
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0154482, 0.0043551, -0.0198125, 0.0198114
1: -0.0392320, 0.0138485, -0.0392090, 0.0138169, -0.0530489, 0.0530575
2: -0.0302696, 0.0302965, -0.0302544, 0.0302679, -0.0605375, 0.0605508
3: -0.0358154, 0.0216712, -0.0357937, 0.0216403, -0.0574557, 0.0574649
4: -0.0284588, 0.0328847, -0.0284418, 0.0328548, -0.0613136, 0.0613265

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125853, upper bound: 0.0128784
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126965, upper bound: 0.0127202
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0132636, 0.0007806, -0.0137383, 0.0009519, -0.0142155, 0.0145189
1: -0.0298148, 0.0070447, -0.0309329, 0.0090424, -0.0388572, 0.0379776
2: -0.0271393, 0.0121166, -0.0286555, 0.0149123, -0.0420516, 0.0407722
3: -0.0289018, 0.0123366, -0.0301543, 0.0142417, -0.0431435, 0.0424909
4: -0.0233742, 0.0143226, -0.0244520, 0.0170415, -0.0404157, 0.0387746

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0122428
time: 0.21 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0127759
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0135726, 0.0009111, -0.0137383, 0.0009519, -0.0145245, 0.0146494
1: -0.0307330, 0.0081123, -0.0309329, 0.0090424, -0.0397755, 0.0390452
2: -0.0281220, 0.0154676, -0.0286555, 0.0149123, -0.0430343, 0.0441231
3: -0.0296205, 0.0134897, -0.0301543, 0.0142417, -0.0438622, 0.0436440
4: -0.0241345, 0.0176071, -0.0244520, 0.0170415, -0.0411760, 0.0420590

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0122428
time: 0.21 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0127759
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0130841, 0.0005465, -0.0160039, 0.0174473
1: -0.0392320, 0.0138485, -0.0291529, 0.0065217, -0.0457537, 0.0430013
2: -0.0302696, 0.0302965, -0.0267945, 0.0107319, -0.0410015, 0.0570910
3: -0.0358154, 0.0216712, -0.0285623, 0.0116865, -0.0475019, 0.0502335
4: -0.0284588, 0.0328847, -0.0228789, 0.0128806, -0.0413394, 0.0557635

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0122665
time: 0.22 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0122665
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154575, 0.0043632, -0.0134172, 0.0006683, -0.0161258, 0.0177804
1: -0.0392320, 0.0138485, -0.0301239, 0.0075661, -0.0467981, 0.0439723
2: -0.0302696, 0.0302965, -0.0278194, 0.0141731, -0.0444427, 0.0581159
3: -0.0358154, 0.0216712, -0.0292925, 0.0128147, -0.0486301, 0.0509637
4: -0.0284588, 0.0328847, -0.0237149, 0.0162432, -0.0447020, 0.0565996

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0129630
time: 0.22 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0129630
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0132636, 0.0007806, -0.0145102, 0.0032983, -0.0165619, 0.0152908
1: -0.0298148, 0.0070447, -0.0354606, 0.0111622, -0.0409769, 0.0425053
2: -0.0271393, 0.0121166, -0.0288524, 0.0250942, -0.0522335, 0.0409690
3: -0.0289018, 0.0123366, -0.0331143, 0.0183202, -0.0472220, 0.0454509
4: -0.0233742, 0.0143226, -0.0265385, 0.0274025, -0.0507767, 0.0408611

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0121344
time: 0.23 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0127052
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0152357, 0.0043167, -0.0145102, 0.0032983, -0.0185340, 0.0188269
1: -0.0386036, 0.0127731, -0.0354606, 0.0111622, -0.0497658, 0.0482338
2: -0.0292743, 0.0297455, -0.0288524, 0.0250942, -0.0543685, 0.0585979
3: -0.0352973, 0.0209170, -0.0331143, 0.0183202, -0.0536175, 0.0540313
4: -0.0277387, 0.0324881, -0.0265385, 0.0274025, -0.0551412, 0.0590267

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126733, upper bound: 0.0121581
time: 0.21 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126733, upper bound: 0.0127289
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0137383, 0.0009519, -0.0132636, 0.0007806, -0.0145189, 0.0142155
1: -0.0309329, 0.0090424, -0.0298148, 0.0070447, -0.0379776, 0.0388572
2: -0.0286555, 0.0149123, -0.0271393, 0.0121166, -0.0407722, 0.0420516
3: -0.0301543, 0.0142417, -0.0289018, 0.0123366, -0.0424909, 0.0431435
4: -0.0244520, 0.0170415, -0.0233742, 0.0143226, -0.0387746, 0.0404157

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0128558
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0128564
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0130841, 0.0005465, -0.0154575, 0.0043632, -0.0174473, 0.0160039
1: -0.0291529, 0.0065217, -0.0392320, 0.0138485, -0.0430013, 0.0457537
2: -0.0267945, 0.0107319, -0.0302696, 0.0302965, -0.0570910, 0.0410015
3: -0.0285623, 0.0116865, -0.0358154, 0.0216712, -0.0502335, 0.0475019
4: -0.0228789, 0.0128806, -0.0284588, 0.0328847, -0.0557635, 0.0413394

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0128772
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0131374
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0134172, 0.0006683, -0.0154575, 0.0043632, -0.0177804, 0.0161258
1: -0.0301239, 0.0075661, -0.0392320, 0.0138485, -0.0439723, 0.0467981
2: -0.0278194, 0.0141731, -0.0302696, 0.0302965, -0.0581159, 0.0444427
3: -0.0292925, 0.0128147, -0.0358154, 0.0216712, -0.0509637, 0.0486301
4: -0.0237149, 0.0162432, -0.0284588, 0.0328847, -0.0565996, 0.0447020

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0125853
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0145102, 0.0032983, -0.0152357, 0.0043167, -0.0188269, 0.0185340
1: -0.0354606, 0.0111622, -0.0386036, 0.0127731, -0.0482338, 0.0497658
2: -0.0288524, 0.0250942, -0.0292743, 0.0297455, -0.0585979, 0.0543685
3: -0.0331143, 0.0183202, -0.0352973, 0.0209170, -0.0540313, 0.0536175
4: -0.0265385, 0.0274025, -0.0277387, 0.0324881, -0.0590267, 0.0551412

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121344, upper bound: 0.0125143
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0121344, upper bound: 0.0125143
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0133644, 0.0004439, -0.0118328, -0.0035663, -0.0097982, 0.0122767
1: -0.0311151, 0.0095714, -0.0206060, 0.0004676, -0.0315826, 0.0301773
2: -0.0292704, 0.0041866, -0.0244539, -0.0027796, -0.0264907, 0.0286405
3: -0.0307164, 0.0145110, -0.0195738, 0.0029438, -0.0336603, 0.0340848
4: -0.0233558, 0.0064767, -0.0203478, -0.0020269, -0.0213290, 0.0268245

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108607, upper bound: 0.0132438
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117492, upper bound: 0.0133389
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0136992, 0.0010155, -0.0118328, -0.0035663, -0.0101330, 0.0128483
1: -0.0310651, 0.0089942, -0.0206060, 0.0004676, -0.0315326, 0.0296001
2: -0.0286015, 0.0155542, -0.0244539, -0.0027796, -0.0258219, 0.0400081
3: -0.0301401, 0.0140987, -0.0195738, 0.0029438, -0.0330839, 0.0336725
4: -0.0244749, 0.0176564, -0.0203478, -0.0020269, -0.0224480, 0.0380042

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0114352, upper bound: 0.0129667
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0109332, upper bound: 0.0132729
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117399, upper bound: 0.0133294
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133644, 0.0004439, -0.0139358, 0.0012009, -0.0145654, 0.0143798
1: -0.0311151, 0.0095714, -0.0315487, 0.0094707, -0.0405858, 0.0411201
2: -0.0292704, 0.0041866, -0.0290280, 0.0158852, -0.0451556, 0.0332146
3: -0.0307164, 0.0145110, -0.0306428, 0.0146923, -0.0454087, 0.0451538
4: -0.0233558, 0.0064767, -0.0248478, 0.0181045, -0.0414603, 0.0313245

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122175, upper bound: 0.0132851
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0124581, upper bound: 0.0134371
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0136992, 0.0010155, -0.0139358, 0.0012009, -0.0149002, 0.0149513
1: -0.0310651, 0.0089942, -0.0315487, 0.0094707, -0.0405358, 0.0405429
2: -0.0286015, 0.0155542, -0.0290280, 0.0158852, -0.0444867, 0.0445822
3: -0.0301401, 0.0140987, -0.0306428, 0.0146923, -0.0448324, 0.0447415
4: -0.0244749, 0.0176564, -0.0248478, 0.0181045, -0.0425794, 0.0425042

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134474
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134474
time: 0.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.29 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0124161
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0127759
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0125853
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0126973
NS_A1_B1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0128784, upper bound: 0.0124398
NS_A1_B1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0126965, upper bound: 0.0127202
NS_A1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0125853, upper bound: 0.0128784
NS_A1_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0126965, upper bound: 0.0127202
NS_A1_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0122428
NS_A1_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124161, upper bound: 0.0127759
NS_A1_B2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0122428
NS_A1_B2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0127759
NS_A1_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0122665
NS_A1_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0122665
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0129630
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0128772, upper bound: 0.0129630
NS_A1_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0121344
NS_A1_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0125041, upper bound: 0.0127052
NS_A1_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0126733, upper bound: 0.0121581
NS_A1_B2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0126733, upper bound: 0.0127289
NS_A2_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0128558
NS_A2_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0128564
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0128772
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0122428, upper bound: 0.0131374
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0125853
NS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0127759, upper bound: 0.0126973
NS_A2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0121344, upper bound: 0.0125143
NS_A2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0121344, upper bound: 0.0125143
NS_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0108607, upper bound: 0.0132438
NS_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0117492, upper bound: 0.0133389
NS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0109332, upper bound: 0.0132729
NS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0117399, upper bound: 0.0133294
NS_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0122175, upper bound: 0.0132851
NS_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0124581, upper bound: 0.0134371
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134474
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.29
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134474

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0152357, 0.0043167, -0.0134172, 0.0006683, -0.0159041, 0.0177340
1: -0.0386036, 0.0127731, -0.0301239, 0.0075661, -0.0461697, 0.0428970
2: -0.0292743, 0.0297455, -0.0278194, 0.0141731, -0.0434474, 0.0575649
3: -0.0352973, 0.0209170, -0.0292925, 0.0128147, -0.0481120, 0.0502095
4: -0.0277387, 0.0324881, -0.0237149, 0.0162432, -0.0439819, 0.0562030

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0150828, 0.0040845, -0.0134172, 0.0006683, -0.0157511, 0.0175017
1: -0.0380876, 0.0124542, -0.0301239, 0.0075661, -0.0456537, 0.0425780
2: -0.0294030, 0.0295660, -0.0278194, 0.0141731, -0.0435761, 0.0573855
3: -0.0346156, 0.0203395, -0.0292925, 0.0128147, -0.0474303, 0.0496320
4: -0.0277026, 0.0320936, -0.0237149, 0.0162432, -0.0439458, 0.0558085

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130841, 0.0005465, -0.0150828, 0.0040845, -0.0171686, 0.0156292
1: -0.0291529, 0.0065217, -0.0380876, 0.0124542, -0.0416070, 0.0446093
2: -0.0267945, 0.0107319, -0.0294030, 0.0295660, -0.0563605, 0.0401349
3: -0.0285623, 0.0116865, -0.0346156, 0.0203395, -0.0489018, 0.0463021
4: -0.0228789, 0.0128806, -0.0277026, 0.0320936, -0.0549725, 0.0405832

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0118328, -0.0035663, -0.0093066, 0.0120510
1: -0.0294580, 0.0081519, -0.0206060, 0.0004676, -0.0299255, 0.0287579
2: -0.0280244, 0.0024157, -0.0244539, -0.0027796, -0.0252448, 0.0268696
3: -0.0294324, 0.0130694, -0.0195738, 0.0029438, -0.0323763, 0.0326432
4: -0.0222233, 0.0049708, -0.0203478, -0.0020269, -0.0201964, 0.0253186

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108565, upper bound: 0.0132438
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108565, upper bound: 0.0132438
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0128742, 0.0000782, -0.0118328, -0.0035663, -0.0093080, 0.0119110
1: -0.0296807, 0.0075168, -0.0206060, 0.0004676, -0.0301483, 0.0281228
2: -0.0279963, 0.0035073, -0.0244539, -0.0027796, -0.0252166, 0.0279612
3: -0.0293004, 0.0125622, -0.0195738, 0.0029438, -0.0322442, 0.0321360
4: -0.0224881, 0.0058042, -0.0203478, -0.0020269, -0.0204612, 0.0261520

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117450, upper bound: 0.0133389
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117450, upper bound: 0.0133389
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0130514, 0.0005897, -0.0118328, -0.0035663, -0.0094851, 0.0124225
1: -0.0292497, 0.0064942, -0.0206060, 0.0004676, -0.0297172, 0.0271001
2: -0.0267610, 0.0114241, -0.0244539, -0.0027796, -0.0239814, 0.0358780
3: -0.0284847, 0.0115453, -0.0195738, 0.0029438, -0.0314286, 0.0311191
4: -0.0228964, 0.0135329, -0.0203478, -0.0020269, -0.0208695, 0.0338807

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108675, upper bound: 0.0132729
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0108675, upper bound: 0.0132729
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0133847, 0.0007248, -0.0118328, -0.0035663, -0.0098185, 0.0125576
1: -0.0303138, 0.0075402, -0.0206060, 0.0004676, -0.0307814, 0.0281461
2: -0.0277657, 0.0148572, -0.0244539, -0.0027796, -0.0249860, 0.0393110
3: -0.0293044, 0.0127149, -0.0195738, 0.0029438, -0.0322482, 0.0322887
4: -0.0237479, 0.0169044, -0.0203478, -0.0020269, -0.0217210, 0.0372522

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117356, upper bound: 0.0133294
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0117356, upper bound: 0.0133294
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0139358, 0.0012009, -0.0140738, 0.0141540
1: -0.0294580, 0.0081519, -0.0315487, 0.0094707, -0.0389287, 0.0397007
2: -0.0280244, 0.0024157, -0.0290280, 0.0158852, -0.0439096, 0.0314437
3: -0.0294324, 0.0130694, -0.0306428, 0.0146923, -0.0441247, 0.0437122
4: -0.0222233, 0.0049708, -0.0248478, 0.0181045, -0.0403278, 0.0298186

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122175, upper bound: 0.0126320
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122175, upper bound: 0.0132851
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0128742, 0.0000782, -0.0139358, 0.0012009, -0.0140752, 0.0140141
1: -0.0296807, 0.0075168, -0.0315487, 0.0094707, -0.0391515, 0.0390656
2: -0.0279963, 0.0035073, -0.0290280, 0.0158852, -0.0438815, 0.0325353
3: -0.0293004, 0.0125622, -0.0306428, 0.0146923, -0.0439927, 0.0432050
4: -0.0224881, 0.0058042, -0.0248478, 0.0181045, -0.0405926, 0.0306520

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123919, upper bound: 0.0127271
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0123919, upper bound: 0.0127271
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0136992, 0.0010155, -0.0133644, 0.0004439, -0.0141432, 0.0143799
1: -0.0310651, 0.0089942, -0.0311151, 0.0095714, -0.0406364, 0.0401093
2: -0.0286015, 0.0155542, -0.0292704, 0.0041866, -0.0327881, 0.0448245
3: -0.0301401, 0.0140987, -0.0307164, 0.0145110, -0.0446511, 0.0448151
4: -0.0244749, 0.0176564, -0.0233558, 0.0064767, -0.0309516, 0.0410122

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126082, upper bound: 0.0125701
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134359
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0136992, 0.0010155, -0.0136992, 0.0010155, -0.0147147, 0.0147147
1: -0.0310651, 0.0089942, -0.0310651, 0.0089942, -0.0400593, 0.0400593
2: -0.0286015, 0.0155542, -0.0286015, 0.0155542, -0.0441557, 0.0441557
3: -0.0301401, 0.0140987, -0.0301401, 0.0140987, -0.0442388, 0.0442388
4: -0.0244749, 0.0176564, -0.0244749, 0.0176564, -0.0421313, 0.0421313

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122150, upper bound: 0.0133146
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134358
time: 0.23 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.50 seconds
NS_A2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0108565, upper bound: 0.0132438
NS_A2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0108565, upper bound: 0.0132438
NS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0117450, upper bound: 0.0133389
NS_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0117450, upper bound: 0.0133389
NS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0108675, upper bound: 0.0132729
NS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0108675, upper bound: 0.0132729
NS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0117356, upper bound: 0.0133294
NS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0117356, upper bound: 0.0133294
NS_A2_B2_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0122175, upper bound: 0.0126320
NS_A2_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0122175, upper bound: 0.0132851
NS_A2_B2_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0123919, upper bound: 0.0127271
NS_A2_B2_A2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0123919, upper bound: 0.0127271
NS_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0126082, upper bound: 0.0125701
NS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134359
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0122150, upper bound: 0.0133146
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.50
Output dim: 0, lower bound: -0.0126145, upper bound: 0.0134358

## BFS NS instance: NS_A2_B2_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0114747, -0.0039066, -0.0089663, 0.0116929
1: -0.0294580, 0.0081519, -0.0202500, 0.0010934, -0.0305513, 0.0284020
2: -0.0280244, 0.0024157, -0.0246436, -0.0062687, -0.0217557, 0.0270593
3: -0.0294324, 0.0130694, -0.0190428, 0.0035940, -0.0330264, 0.0321122
4: -0.0222233, 0.0049708, -0.0196523, -0.0045450, -0.0176783, 0.0246231

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0117075, -0.0037467, -0.0091262, 0.0119256
1: -0.0294580, 0.0081519, -0.0202905, 0.0001932, -0.0296511, 0.0284425
2: -0.0280244, 0.0024157, -0.0242052, -0.0029088, -0.0251157, 0.0266209
3: -0.0294324, 0.0130694, -0.0192134, 0.0025881, -0.0320205, 0.0322828
4: -0.0222233, 0.0049708, -0.0201841, -0.0022380, -0.0199853, 0.0251549

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0128742, 0.0000782, -0.0114747, -0.0039066, -0.0089677, 0.0115529
1: -0.0296807, 0.0075168, -0.0202500, 0.0010934, -0.0307741, 0.0277668
2: -0.0279963, 0.0035073, -0.0246436, -0.0062687, -0.0217276, 0.0281509
3: -0.0293004, 0.0125622, -0.0190428, 0.0035940, -0.0328943, 0.0316050
4: -0.0224881, 0.0058042, -0.0196523, -0.0045450, -0.0179431, 0.0254564

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0128742, 0.0000782, -0.0117075, -0.0037467, -0.0091276, 0.0117857
1: -0.0296807, 0.0075168, -0.0202905, 0.0001932, -0.0298739, 0.0278073
2: -0.0279963, 0.0035073, -0.0242052, -0.0029088, -0.0250875, 0.0277125
3: -0.0293004, 0.0125622, -0.0192134, 0.0025881, -0.0318885, 0.0317756
4: -0.0224881, 0.0058042, -0.0201841, -0.0022380, -0.0202501, 0.0259883

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0130514, 0.0005897, -0.0114747, -0.0039066, -0.0091448, 0.0120644
1: -0.0292497, 0.0064942, -0.0202500, 0.0010934, -0.0303431, 0.0267442
2: -0.0267610, 0.0114241, -0.0246436, -0.0062687, -0.0204923, 0.0360677
3: -0.0284847, 0.0115453, -0.0190428, 0.0035940, -0.0320787, 0.0305881
4: -0.0228964, 0.0135329, -0.0196523, -0.0045450, -0.0183514, 0.0331852

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130514, 0.0005897, -0.0117075, -0.0037467, -0.0093047, 0.0122972
1: -0.0292497, 0.0064942, -0.0202905, 0.0001932, -0.0294428, 0.0267847
2: -0.0267610, 0.0114241, -0.0242052, -0.0029088, -0.0238522, 0.0356293
3: -0.0284847, 0.0115453, -0.0192134, 0.0025881, -0.0310729, 0.0307587
4: -0.0228964, 0.0135329, -0.0201841, -0.0022380, -0.0206584, 0.0337170

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0133847, 0.0007248, -0.0114747, -0.0039066, -0.0094781, 0.0121995
1: -0.0303138, 0.0075402, -0.0202500, 0.0010934, -0.0314072, 0.0277902
2: -0.0277657, 0.0148572, -0.0246436, -0.0062687, -0.0214970, 0.0395007
3: -0.0293044, 0.0127149, -0.0190428, 0.0035940, -0.0328983, 0.0317577
4: -0.0237479, 0.0169044, -0.0196523, -0.0045450, -0.0192029, 0.0365567

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133847, 0.0007248, -0.0117075, -0.0037467, -0.0096380, 0.0124323
1: -0.0303138, 0.0075402, -0.0202905, 0.0001932, -0.0305070, 0.0278307
2: -0.0277657, 0.0148572, -0.0242052, -0.0029088, -0.0248569, 0.0390623
3: -0.0293044, 0.0127149, -0.0192134, 0.0025881, -0.0318925, 0.0319283
4: -0.0237479, 0.0169044, -0.0201841, -0.0022380, -0.0215099, 0.0370885

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0135980, 0.0009096, -0.0137825, 0.0138162
1: -0.0294580, 0.0081519, -0.0307039, 0.0080062, -0.0374641, 0.0388559
2: -0.0280244, 0.0024157, -0.0281823, 0.0151782, -0.0432026, 0.0305980
3: -0.0294324, 0.0130694, -0.0297287, 0.0132973, -0.0427297, 0.0427981
4: -0.0222233, 0.0049708, -0.0241084, 0.0173418, -0.0395651, 0.0290793

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122165, upper bound: 0.0132851
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122164, upper bound: 0.0132851
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0136992, 0.0010155, -0.0128742, 0.0000782, -0.0137775, 0.0138897
1: -0.0310651, 0.0089942, -0.0296807, 0.0075168, -0.0385819, 0.0386749
2: -0.0286015, 0.0155542, -0.0279963, 0.0035073, -0.0321088, 0.0435504
3: -0.0301401, 0.0140987, -0.0293004, 0.0125622, -0.0427023, 0.0433991
4: -0.0244749, 0.0176564, -0.0224881, 0.0058042, -0.0302791, 0.0401445

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122150, upper bound: 0.0133205
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122150, upper bound: 0.0133205
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0130514, 0.0005897, -0.0136992, 0.0010155, -0.0140669, 0.0142890
1: -0.0292497, 0.0064942, -0.0310651, 0.0089942, -0.0382439, 0.0375592
2: -0.0267610, 0.0114241, -0.0286015, 0.0155542, -0.0423152, 0.0400256
3: -0.0284847, 0.0115453, -0.0301401, 0.0140987, -0.0425834, 0.0416854
4: -0.0228964, 0.0135329, -0.0244749, 0.0176564, -0.0405528, 0.0380079

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0122253, upper bound: 0.0126237
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0122253, upper bound: 0.0133145
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133847, 0.0007248, -0.0136992, 0.0010155, -0.0144002, 0.0144241
1: -0.0303138, 0.0075402, -0.0310651, 0.0089942, -0.0393080, 0.0386052
2: -0.0277657, 0.0148572, -0.0286015, 0.0155542, -0.0433198, 0.0434587
3: -0.0293044, 0.0127149, -0.0301401, 0.0140987, -0.0434031, 0.0428550
4: -0.0237479, 0.0169044, -0.0244749, 0.0176564, -0.0414043, 0.0413793

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126082, upper bound: 0.0126277
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0126082, upper bound: 0.0126277
time: 0.24 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.53 seconds
NS_A2_B2_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0122165, upper bound: 0.0132851
NS_A2_B2_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0122164, upper bound: 0.0132851
NS_A2_B2_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0122150, upper bound: 0.0133205
NS_A2_B2_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0122150, upper bound: 0.0133205
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0122253, upper bound: 0.0126237
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0122253, upper bound: 0.0133145
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0126082, upper bound: 0.0126277
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.53
Output dim: 0, lower bound: -0.0126082, upper bound: 0.0126277

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0128391, 0.0000782, -0.0129511, 0.0130572
1: -0.0294580, 0.0081519, -0.0295072, 0.0075168, -0.0369748, 0.0376592
2: -0.0280244, 0.0024157, -0.0279899, 0.0035073, -0.0315317, 0.0304056
3: -0.0294324, 0.0130694, -0.0291544, 0.0125622, -0.0419946, 0.0422238
4: -0.0222233, 0.0049708, -0.0224828, 0.0058042, -0.0280275, 0.0274537

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0128728, 0.0002182, -0.0133847, 0.0007248, -0.0135977, 0.0136029
1: -0.0294580, 0.0081519, -0.0303138, 0.0075402, -0.0369981, 0.0384657
2: -0.0280244, 0.0024157, -0.0277657, 0.0148572, -0.0428816, 0.0301814
3: -0.0294324, 0.0130694, -0.0293044, 0.0127149, -0.0421473, 0.0423738
4: -0.0222233, 0.0049708, -0.0237479, 0.0169044, -0.0391277, 0.0287187

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0130514, 0.0005897, -0.0128742, 0.0000782, -0.0131296, 0.0134640
1: -0.0292497, 0.0064942, -0.0296807, 0.0075168, -0.0367665, 0.0361749
2: -0.0267610, 0.0114241, -0.0279963, 0.0035073, -0.0302683, 0.0394204
3: -0.0284847, 0.0115453, -0.0293004, 0.0125622, -0.0410469, 0.0408457
4: -0.0228964, 0.0135329, -0.0224881, 0.0058042, -0.0287005, 0.0360210

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133847, 0.0007248, -0.0128742, 0.0000782, -0.0134629, 0.0135991
1: -0.0303138, 0.0075402, -0.0296807, 0.0075168, -0.0378306, 0.0372209
2: -0.0277657, 0.0148572, -0.0279963, 0.0035073, -0.0312730, 0.0428534
3: -0.0293044, 0.0127149, -0.0293004, 0.0125622, -0.0418666, 0.0420152
4: -0.0237479, 0.0169044, -0.0224881, 0.0058042, -0.0295520, 0.0393925

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0130514, 0.0005897, -0.0133847, 0.0007248, -0.0137762, 0.0139744
1: -0.0292497, 0.0064942, -0.0303138, 0.0075402, -0.0367898, 0.0368080
2: -0.0267610, 0.0114241, -0.0277657, 0.0148572, -0.0416181, 0.0391898
3: -0.0284847, 0.0115453, -0.0293044, 0.0127149, -0.0411996, 0.0408497
4: -0.0228964, 0.0135329, -0.0237479, 0.0169044, -0.0398008, 0.0372808

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 38

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.45 + 89.86 = 91.32 seconds
