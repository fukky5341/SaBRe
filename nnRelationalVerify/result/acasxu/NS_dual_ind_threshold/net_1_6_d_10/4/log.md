## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.198190501800003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.9076786, 12.5877399, -4.9076786, 12.5877399, -17.4954185, 17.4954185)
1: (-13.2235098, 19.0510464, -13.2235098, 19.0510464, -32.2745552, 32.2745476)
2: (-8.4530544, 17.2152138, -8.4530544, 17.2152138, -25.6682663, 25.6682663)
3: (-14.9454336, 21.4810410, -14.9454336, 21.4810410, -36.4264717, 36.4264717)
4: (-10.1239004, 20.9261990, -10.1239004, 20.9261990, -31.0500984, 31.0500984)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.80 + 1.89 = 2.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -27.2117964, upper bound: 27.2117964

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2096536, upper bound: 27.2091887
time: 0.79 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091721, upper bound: 27.2091721
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.75 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -27.2096536, upper bound: 27.2091887
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -27.2091721, upper bound: 27.2091721

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.5904746, 11.7819414, -4.9076786, 12.5877399, -17.1782150, 16.6896172
1: -12.4367714, 17.9547424, -13.2235098, 19.0510464, -31.4878159, 31.1782532
2: -7.9547815, 16.1541290, -8.4530544, 17.2152138, -25.1699944, 24.6071835
3: -14.0239172, 20.1486168, -14.9454336, 21.4810410, -35.5049591, 35.0940514
4: -9.5049162, 19.6281281, -10.1239004, 20.9261990, -30.4311142, 29.7520294

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
time: 0.58 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
time: 0.99 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.6723428, 11.9120274, -4.8962965, 12.5573883, -17.2297306, 16.8083229
1: -12.5596266, 18.0415668, -13.1924715, 19.0064507, -31.5660706, 31.2340374
2: -8.0017109, 16.3343449, -8.4328079, 17.1745396, -25.1762486, 24.7671490
3: -14.2002497, 20.3525009, -14.9100819, 21.4296112, -35.6298599, 35.2625771
4: -9.5917578, 19.8698196, -10.0992765, 20.8776894, -30.4694481, 29.9690971

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091721
time: 0.77 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091721
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091694
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091721
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 4, lower bound: -27.2091694, upper bound: 27.2091721

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.5904746, 11.7819414, -4.5904746, 11.7819414, -16.3724117, 16.3724136
1: -12.4367714, 17.9547424, -12.4367714, 17.9547424, -30.3915100, 30.3915100
2: -7.9547815, 16.1541290, -7.9547815, 16.1541290, -24.1089096, 24.1089096
3: -14.0239172, 20.1486168, -14.0239172, 20.1486168, -34.1725349, 34.1725349
4: -9.5049162, 19.6281281, -9.5049162, 19.6281281, -29.1330452, 29.1330452

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2096226, upper bound: 27.2072118
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2096509, upper bound: 27.2091887
time: 0.90 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.5904746, 11.7819414, -4.6723428, 11.9120274, -16.5025024, 16.4542809
1: -12.4367714, 17.9547424, -12.5596266, 18.0415668, -30.4783363, 30.5143661
2: -7.9547815, 16.1541290, -8.0017109, 16.3343449, -24.2891254, 24.1558380
3: -14.0239172, 20.1486168, -14.2002497, 20.3525009, -34.3764191, 34.3488655
4: -9.5049162, 19.6281281, -9.5917578, 19.8698196, -29.3747368, 29.2198868

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2096226, upper bound: 27.2072118
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2096226, upper bound: 27.2091887
time: 0.56 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.6723428, 11.9120274, -4.5904746, 11.7819414, -16.4542809, 16.5025024
1: -12.5596266, 18.0415668, -12.4367714, 17.9547424, -30.5143661, 30.4783363
2: -8.0017109, 16.3343449, -7.9547815, 16.1541290, -24.1558380, 24.2891254
3: -14.2002497, 20.3525009, -14.0239172, 20.1486168, -34.3488655, 34.3764191
4: -9.5917578, 19.8698196, -9.5049162, 19.6281281, -29.2198868, 29.3747368

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2090673, upper bound: 27.2081852
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2086284, upper bound: 27.2086311
time: 0.55 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.6723428, 11.9120274, -4.6723428, 11.9120274, -16.5843678, 16.5843678
1: -12.5596266, 18.0415668, -12.5596266, 18.0415668, -30.6011906, 30.6011925
2: -8.0017109, 16.3343449, -8.0017109, 16.3343449, -24.3360538, 24.3360538
3: -14.2002497, 20.3525009, -14.2002497, 20.3525009, -34.5527458, 34.5527458
4: -9.5917578, 19.8698196, -9.5917578, 19.8698196, -29.4615784, 29.4615784

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2090673, upper bound: 27.2081852
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2086284, upper bound: 27.2086311
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.73 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2096226, upper bound: 27.2072118
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2096509, upper bound: 27.2091887
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2096226, upper bound: 27.2072118
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2096226, upper bound: 27.2091887
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2090673, upper bound: 27.2081852
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2086284, upper bound: 27.2086311
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2090673, upper bound: 27.2081852
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 4, lower bound: -27.2086284, upper bound: 27.2086311

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.2093120, 10.7124605, -4.5056977, 11.5624132, -15.7717247, 15.2181587
1: -11.3399172, 16.3705730, -12.2098236, 17.6363068, -28.9762230, 28.5803947
2: -7.2677803, 14.6629076, -7.8075700, 15.8582945, -23.1260757, 22.4704742
3: -12.7595739, 18.3373203, -13.7655125, 19.7816315, -32.5412064, 32.1028252
4: -8.6866646, 17.8167229, -9.3272247, 19.2698040, -27.9564686, 27.1439457

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2007912, upper bound: 27.1970684
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2095701, upper bound: 27.2072076
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.4492803, 11.4000769, -4.5904746, 11.7819414, -16.2312183, 15.9905510
1: -12.0554895, 17.3962154, -12.4367714, 17.9547424, -30.0102291, 29.8329868
2: -7.7079463, 15.6410189, -7.9547815, 16.1541290, -23.8620758, 23.5957966
3: -13.5865173, 19.5068531, -14.0239172, 20.1486168, -33.7351341, 33.5307693
4: -9.2065659, 19.0087242, -9.5049162, 19.6281281, -28.8346939, 28.5136414

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2076933, upper bound: 27.2096419
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2076933, upper bound: 27.2096702
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.2093120, 10.7124605, -4.5884914, 11.6938124, -15.9031239, 15.3009520
1: -11.3399172, 16.3705730, -12.3350935, 17.7242413, -29.0641575, 28.7056618
2: -7.2677803, 14.6629076, -7.8549924, 16.0430183, -23.3107948, 22.5179005
3: -12.7595739, 18.3373203, -13.9441872, 19.9862003, -32.7457733, 32.2815056
4: -8.6866646, 17.8167229, -9.4144096, 19.5178623, -28.2045269, 27.2311306

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2071097
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2089638, upper bound: 27.2066708
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.4492803, 11.4000769, -4.6723428, 11.9120274, -16.3613052, 16.0724201
1: -12.0554895, 17.3962154, -12.5596266, 18.0415668, -30.0970573, 29.9558411
2: -7.7079463, 15.6410189, -8.0017109, 16.3343449, -24.0422916, 23.6427269
3: -13.5865173, 19.5068531, -14.2002497, 20.3525009, -33.9390144, 33.7070961
4: -9.2065659, 19.0087242, -9.5917578, 19.8698196, -29.0763855, 28.6004829

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2090866
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2089638, upper bound: 27.2086478
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.6073575, 11.7263937, -4.5904746, 11.7819414, -16.3892956, 16.3168678
1: -12.3935480, 17.7883129, -12.4367714, 17.9547424, -30.3482876, 30.2250786
2: -7.8905721, 16.0862961, -7.9547815, 16.1541290, -24.0447006, 24.0410748
3: -14.0091572, 20.0458050, -14.0239172, 20.1486168, -34.1577759, 34.0697212
4: -9.4518366, 19.5683060, -9.5049162, 19.6281281, -29.0799637, 29.0732231

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2086578
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2086578
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.6157975, 11.7587185, -4.5904746, 11.7819414, -16.3977375, 16.3491936
1: -12.4065723, 17.8174953, -12.4367714, 17.9547424, -30.3613129, 30.2542667
2: -7.9025259, 16.1275082, -7.9547815, 16.1541290, -24.0566559, 24.0822887
3: -14.0258732, 20.0947495, -14.0239172, 20.1486168, -34.1744919, 34.1186676
4: -9.4722614, 19.6183720, -9.5049162, 19.6281281, -29.1003895, 29.1232872

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2091034
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2091034
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.6073575, 11.7263937, -4.6723428, 11.9120274, -16.5193844, 16.3987331
1: -12.3935480, 17.7883129, -12.5596266, 18.0415668, -30.4351139, 30.3479328
2: -7.8905721, 16.0862961, -8.0017109, 16.3343449, -24.2249165, 24.0880051
3: -14.0091572, 20.0458050, -14.2002497, 20.3525009, -34.3616562, 34.2460556
4: -9.4518366, 19.5683060, -9.5917578, 19.8698196, -29.3216553, 29.1600647

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2081852
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2081852
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.6157975, 11.7587185, -4.6723428, 11.9120274, -16.5278244, 16.4310608
1: -12.4065723, 17.8174953, -12.5596266, 18.0415668, -30.4481392, 30.3771210
2: -7.9025259, 16.1275082, -8.0017109, 16.3343449, -24.2368698, 24.1292171
3: -14.0258732, 20.0947495, -14.2002497, 20.3525009, -34.3783722, 34.2949944
4: -9.4722614, 19.6183720, -9.5917578, 19.8698196, -29.3420811, 29.2101288

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2086311
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2086311
time: 0.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.36 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2007912, upper bound: 27.1970684
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2095701, upper bound: 27.2072076
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2076933, upper bound: 27.2096419
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2076933, upper bound: 27.2096702
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2071097
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2089638, upper bound: 27.2066708
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2090866
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2089638, upper bound: 27.2086478
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2086578
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2086578
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2091034
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2083526, upper bound: 27.2091034
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2081852
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2081852
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2086311
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 4, lower bound: -27.2081852, upper bound: 27.2086311

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.2093120, 10.7124605, -4.2110252, 10.8017759, -15.0110874, 14.9234858
1: -11.3399172, 16.3705730, -11.3982258, 16.4986095, -27.8385258, 27.7687950
2: -7.2677803, 14.6629076, -7.2841597, 14.8236675, -22.0914459, 21.9470634
3: -12.7595739, 18.3373203, -12.8505898, 18.4896584, -31.2492332, 31.1879101
4: -8.6866646, 17.8167229, -8.7049274, 18.0290489, -26.7157135, 26.5216503

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1966358
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.2093120, 10.7124605, -4.4423351, 11.3971930, -15.6065044, 15.1547956
1: -11.3399172, 16.3705730, -12.0408401, 17.3948021, -28.7347183, 28.4114113
2: -7.2677803, 14.6629076, -7.6988130, 15.6351929, -22.9029732, 22.3617153
3: -12.7595739, 18.3373203, -13.5728292, 19.5053501, -32.2649231, 31.9101467
4: -8.6866646, 17.8167229, -9.1967487, 18.9991550, -27.6858196, 27.0134697

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2085488, upper bound: 27.2067750
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2085488, upper bound: 27.2072076
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.4492803, 11.4000769, -4.2093120, 10.7124605, -15.1617384, 15.6093884
1: -12.0554895, 17.3962154, -11.3399172, 16.3705730, -28.4260616, 28.7361317
2: -7.7079463, 15.6410189, -7.2677803, 14.6629076, -22.3708534, 22.9087963
3: -13.5865173, 19.5068531, -12.7595739, 18.3373203, -31.9238358, 32.2664261
4: -9.2065659, 19.0087242, -8.6866646, 17.8167229, -27.0232887, 27.6953888

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1970684, upper bound: 27.2007912
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2072076, upper bound: 27.2095701
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.4492803, 11.4000769, -4.4492803, 11.4000769, -15.8493576, 15.8493576
1: -12.0554895, 17.3962154, -12.0554895, 17.3962154, -29.4517059, 29.4517059
2: -7.7079463, 15.6410189, -7.7079463, 15.6410189, -23.3489647, 23.3489647
3: -13.5865173, 19.5068531, -13.5865173, 19.5068531, -33.0933685, 33.0933685
4: -9.2065659, 19.0087242, -9.2065659, 19.0087242, -28.2152901, 28.2152901

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1970684, upper bound: 27.2007912
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2072076, upper bound: 27.2095984
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.2093120, 10.7124605, -4.5307307, 11.5260963, -15.7354088, 15.2431908
1: -11.3399172, 16.3705730, -12.1875830, 17.4963818, -28.8362980, 28.5581551
2: -7.2677803, 14.6629076, -7.7560701, 15.8192520, -23.0870304, 22.4189777
3: -12.7595739, 18.3373203, -13.7741108, 19.7097473, -32.4693222, 32.1114311
4: -8.6866646, 17.8167229, -9.2893763, 19.2453976, -27.9320621, 27.1060982

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2066708
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2066708
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.2093120, 10.7124605, -4.5322189, 11.5413418, -15.7506542, 15.2446766
1: -11.3399172, 16.3705730, -12.1827869, 17.5013218, -28.8412399, 28.5533600
2: -7.2677803, 14.6629076, -7.7563672, 15.8373184, -23.1050987, 22.4192734
3: -12.7595739, 18.3373203, -13.7707005, 19.7298737, -32.4894485, 32.1080093
4: -8.6866646, 17.8167229, -9.2955933, 19.2678051, -27.9544697, 27.1123123

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2089638, upper bound: 27.2066708
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2089611, upper bound: 27.2066708
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.4492803, 11.4000769, -4.6073575, 11.7263937, -16.1756725, 16.0074348
1: -12.0554895, 17.3962154, -12.3935480, 17.7883129, -29.8438015, 29.7897625
2: -7.7079463, 15.6410189, -7.8905721, 16.0862961, -23.7942429, 23.5315914
3: -13.5865173, 19.5068531, -14.0091572, 20.0458050, -33.6323242, 33.5160103
4: -9.2065659, 19.0087242, -9.4518366, 19.5683060, -28.7748718, 28.4605598

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2086578, upper bound: 27.2083526
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2086578, upper bound: 27.2086478
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.4492803, 11.4000769, -4.6157975, 11.7587185, -16.2079983, 16.0158749
1: -12.0554895, 17.3962154, -12.4065723, 17.8174953, -29.8729858, 29.8027878
2: -7.7079463, 15.6410189, -7.9025259, 16.1275082, -23.8354549, 23.5435448
3: -13.5865173, 19.5068531, -14.0258732, 20.0947495, -33.6812630, 33.5327263
4: -9.2065659, 19.0087242, -9.4722614, 19.6183720, -28.8249378, 28.4809837

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2089611, upper bound: 27.2083526
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2091034, upper bound: 27.2086478
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.6073575, 11.7263937, -4.5331025, 11.6178961, -16.2252541, 16.2594967
1: -12.3935480, 17.7883129, -12.2875996, 17.7315407, -30.1250858, 30.0759125
2: -7.8905721, 16.0862961, -7.8551126, 15.9343672, -23.8249397, 23.9414082
3: -14.0091572, 20.0458050, -13.8514175, 19.8753357, -33.8844910, 33.8972244
4: -9.4518366, 19.5683060, -9.3789968, 19.3634605, -28.8152962, 28.9473038

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2065092
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.6073575, 11.7263937, -4.5361848, 11.6357536, -16.2431107, 16.2625771
1: -12.3935480, 17.7883129, -12.2895918, 17.7398758, -30.1334229, 30.0778999
2: -7.8905721, 16.0862961, -7.8600860, 15.9560280, -23.8465996, 23.9463825
3: -14.0091572, 20.0458050, -13.8556681, 19.9008865, -33.9100418, 33.9014740
4: -9.4518366, 19.5683060, -9.3909245, 19.3876572, -28.8394928, 28.9592285

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2065092
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.6157975, 11.7587185, -4.5331025, 11.6178961, -16.2336941, 16.2918205
1: -12.4065723, 17.8174953, -12.2875996, 17.7315407, -30.1381111, 30.1050949
2: -7.9025259, 16.1275082, -7.8551126, 15.9343672, -23.8368931, 23.9826202
3: -14.0258732, 20.0947495, -13.8514175, 19.8753357, -33.9012070, 33.9461670
4: -9.4722614, 19.6183720, -9.3789968, 19.3634605, -28.8357201, 28.9973679

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2077200, upper bound: 27.2069552
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.6157975, 11.7587185, -4.5361848, 11.6357536, -16.2515507, 16.2949028
1: -12.4065723, 17.8174953, -12.2895918, 17.7398758, -30.1464481, 30.1070862
2: -7.9025259, 16.1275082, -7.8600860, 15.9560280, -23.8585548, 23.9875946
3: -14.0258732, 20.0947495, -13.8556681, 19.9008865, -33.9267578, 33.9504128
4: -9.4722614, 19.6183720, -9.3909245, 19.3876572, -28.8599186, 29.0092945

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2077200, upper bound: 27.2069552
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.6073575, 11.7263937, -4.6073575, 11.7263937, -16.3337517, 16.3337517
1: -12.3935480, 17.7883129, -12.3935480, 17.7883129, -30.1818581, 30.1818542
2: -7.8905721, 16.0862961, -7.8905721, 16.0862961, -23.9768677, 23.9768677
3: -14.0091572, 20.0458050, -14.0091572, 20.0458050, -34.0549622, 34.0549622
4: -9.4518366, 19.5683060, -9.4518366, 19.5683060, -29.0201416, 29.0201416

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2079117, upper bound: 27.2063483
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081070, upper bound: 27.2063151
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.6073575, 11.7263937, -4.6157975, 11.7587185, -16.3660755, 16.3421917
1: -12.3935480, 17.7883129, -12.4065723, 17.8174953, -30.2110424, 30.1948853
2: -7.8905721, 16.0862961, -7.9025259, 16.1275082, -24.0180798, 23.9888229
3: -14.0091572, 20.0458050, -14.0258732, 20.0947495, -34.1039047, 34.0716782
4: -9.4518366, 19.5683060, -9.4722614, 19.6183720, -29.0702076, 29.0405674

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2079117, upper bound: 27.2063483
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081070, upper bound: 27.2063151
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.6157975, 11.7587185, -4.6073575, 11.7263937, -16.3421917, 16.3660755
1: -12.4065723, 17.8174953, -12.3935480, 17.7883129, -30.1948853, 30.2110443
2: -7.9025259, 16.1275082, -7.8905721, 16.0862961, -23.9888229, 24.0180798
3: -14.0258732, 20.0947495, -14.0091572, 20.0458050, -34.0716782, 34.1039047
4: -9.4722614, 19.6183720, -9.4518366, 19.5683060, -29.0405674, 29.0702076

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2041384, upper bound: 27.2041970
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2036973, upper bound: 27.2040520
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.6157975, 11.7587185, -4.6157975, 11.7587185, -16.3745155, 16.3745155
1: -12.4065723, 17.8174953, -12.4065723, 17.8174953, -30.2240677, 30.2240677
2: -7.9025259, 16.1275082, -7.9025259, 16.1275082, -24.0300331, 24.0300331
3: -14.0258732, 20.0947495, -14.0258732, 20.0947495, -34.1206207, 34.1206207
4: -9.4722614, 19.6183720, -9.4722614, 19.6183720, -29.0906315, 29.0906315

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2041385, upper bound: 27.2042381
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2036973, upper bound: 27.2041884
time: 1.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.97 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1966358
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2085488, upper bound: 27.2067750
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2085488, upper bound: 27.2072076
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.1970684, upper bound: 27.2007912
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2072076, upper bound: 27.2095701
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.1970684, upper bound: 27.2007912
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2072076, upper bound: 27.2095984
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2066708
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2085178, upper bound: 27.2066708
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2089638, upper bound: 27.2066708
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2089611, upper bound: 27.2066708
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2086578, upper bound: 27.2083526
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2086578, upper bound: 27.2086478
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2089611, upper bound: 27.2083526
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2091034, upper bound: 27.2086478
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2065092
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2065092
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2077200, upper bound: 27.2069552
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2077200, upper bound: 27.2069552
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2079117, upper bound: 27.2063483
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2081070, upper bound: 27.2063151
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2079117, upper bound: 27.2063483
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2081070, upper bound: 27.2063151
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2041384, upper bound: 27.2041970
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2036973, upper bound: 27.2040520
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2041385, upper bound: 27.2042381
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.97
Output dim: 4, lower bound: -27.2036973, upper bound: 27.2041884

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9416430, 10.0241356, -4.2110252, 10.8017759, -14.7434187, 14.2351599
1: -10.6106949, 15.3450384, -11.3982258, 16.4986095, -27.1093044, 26.7432632
2: -6.7955890, 13.7306890, -7.2841597, 14.8236675, -21.6192570, 21.0148487
3: -11.9359779, 17.1758213, -12.8505898, 18.4896584, -30.4256325, 30.0264111
4: -8.1233921, 16.6995983, -8.7049274, 18.0290489, -26.1524410, 25.4045258

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1966358
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1966358
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1433077, 10.5423946, -4.2110252, 10.8017759, -14.9450836, 14.7534189
1: -11.1630354, 16.1227474, -11.3982258, 16.4986095, -27.6616402, 27.5209675
2: -7.1534371, 14.4338942, -7.2841597, 14.8236675, -21.9771004, 21.7180538
3: -12.5585098, 18.0531578, -12.8505898, 18.4896584, -31.0481682, 30.9037476
4: -8.5494747, 17.5393734, -8.7049274, 18.0290489, -26.5785236, 26.2443008

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9416430, 10.0241356, -4.4423351, 11.3971930, -15.3388357, 14.4664707
1: -10.6106949, 15.3450384, -12.0408401, 17.3948021, -28.0054970, 27.3858795
2: -6.7955890, 13.7306890, -7.6988130, 15.6351929, -22.4307823, 21.4295006
3: -11.9359779, 17.1758213, -13.5728292, 19.5053501, -31.4413223, 30.7486496
4: -8.1233921, 16.6995983, -9.1967487, 18.9991550, -27.1225433, 25.8963470

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2067467
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2067750
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1433077, 10.5423946, -4.4423351, 11.3971930, -15.5405006, 14.9847298
1: -11.1630354, 16.1227474, -12.0408401, 17.3948021, -28.5578308, 28.1635838
2: -7.1534371, 14.4338942, -7.6988130, 15.6351929, -22.7886257, 22.1327057
3: -12.5585098, 18.0531578, -13.5728292, 19.5053501, -32.0638542, 31.6259823
4: -8.5494747, 17.5393734, -9.1967487, 18.9991550, -27.5486298, 26.7361202

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2070811
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2070811
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1428089, 10.6177177, -4.2093120, 10.7124605, -14.8552694, 14.8270292
1: -11.2123156, 16.2294140, -11.3399172, 16.3705730, -27.5828857, 27.5693302
2: -7.1643281, 14.5773115, -7.2677803, 14.6629076, -21.8272362, 21.8450890
3: -12.6359673, 18.1798058, -12.7595739, 18.3373203, -30.9732857, 30.9393806
4: -8.5602360, 17.7325573, -8.6866646, 17.8167229, -26.3769569, 26.4192219

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966358, upper bound: 27.1997698
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966358, upper bound: 27.2007912
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3870020, 11.2383909, -4.2093120, 10.7124605, -15.0994606, 15.4477024
1: -11.8893881, 17.1601810, -11.3399172, 16.3705730, -28.2599602, 28.5000973
2: -7.6013036, 15.4227715, -7.2677803, 14.6629076, -22.2642059, 22.6905479
3: -13.3972139, 19.2366486, -12.7595739, 18.3373203, -31.7345314, 31.9962215
4: -9.0787373, 18.7438374, -8.6866646, 17.8167229, -26.8954582, 27.4305019

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2067467, upper bound: 27.2085488
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2067467, upper bound: 27.2095701
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1428089, 10.6177177, -4.4492803, 11.4000769, -15.5428858, 15.0669956
1: -11.2123156, 16.2294140, -12.0554895, 17.3962154, -28.6085300, 28.2849007
2: -7.1643281, 14.5773115, -7.7079463, 15.6410189, -22.8053474, 22.2852554
3: -12.6359673, 18.1798058, -13.5865173, 19.5068531, -32.1428185, 31.7663212
4: -8.5602360, 17.7325573, -9.2065659, 19.0087242, -27.5689583, 26.9391232

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1906803, upper bound: 27.1906803
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1906803, upper bound: 27.2007912
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3870020, 11.2383909, -4.4492803, 11.4000769, -15.7870741, 15.6876707
1: -11.8893881, 17.1601810, -12.0554895, 17.3962154, -29.2856007, 29.2156696
2: -7.6013036, 15.4227715, -7.7079463, 15.6410189, -23.2423172, 23.1307163
3: -13.3972139, 19.2366486, -13.5865173, 19.5068531, -32.9040642, 32.8231659
4: -9.0787373, 18.7438374, -9.2065659, 19.0087242, -28.0874596, 27.9504032

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2008195, upper bound: 27.1991030
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2008195, upper bound: 27.2095984
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.1747007, 10.5993958, -4.5307307, 11.5260963, -15.7007971, 15.1301250
1: -11.2539225, 16.2166843, -12.1875830, 17.4963818, -28.7503052, 28.4042664
2: -7.2063785, 14.5116491, -7.7560701, 15.8192520, -23.0256310, 22.2677193
3: -12.6583109, 18.1518974, -13.7741108, 19.7097473, -32.3680573, 31.9260082
4: -8.6068707, 17.6327438, -9.2893763, 19.2453976, -27.8522682, 26.9221191

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1557441, 10.5723639, -4.5307307, 11.5260963, -15.6818409, 15.1030941
1: -11.1955223, 16.1672478, -12.1875830, 17.4963818, -28.6919022, 28.3548298
2: -7.1743021, 14.4743662, -7.7560701, 15.8192520, -22.9935532, 22.2304363
3: -12.5945053, 18.1012173, -13.7741108, 19.7097473, -32.3042526, 31.8753223
4: -8.5741777, 17.5883846, -9.2893763, 19.2453976, -27.8195744, 26.8777618

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.1747007, 10.5993958, -4.5322189, 11.5413418, -15.7160425, 15.1316128
1: -11.2539225, 16.2166843, -12.1827869, 17.5013218, -28.7552452, 28.3994713
2: -7.2063785, 14.5116491, -7.7563672, 15.8373184, -23.0436974, 22.2680168
3: -12.6583109, 18.1518974, -13.7707005, 19.7298737, -32.3881836, 31.9225883
4: -8.6068707, 17.6327438, -9.2955933, 19.2678051, -27.8746758, 26.9283352

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1557441, 10.5723639, -4.5322189, 11.5413418, -15.6970863, 15.1045818
1: -11.1955223, 16.1672478, -12.1827869, 17.5013218, -28.6968441, 28.3500328
2: -7.1743021, 14.4743662, -7.7563672, 15.8373184, -23.0116196, 22.2307339
3: -12.5945053, 18.1012173, -13.7707005, 19.7298737, -32.3243752, 31.8719101
4: -8.5741777, 17.5883846, -9.2955933, 19.2678051, -27.8419838, 26.8839741

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.3901463, 11.2314816, -4.6073575, 11.7263937, -16.1165390, 15.8388386
1: -11.9016247, 17.1665440, -12.3935480, 17.7883129, -29.6899357, 29.5600891
2: -7.6050115, 15.4159374, -7.8905721, 16.0862961, -23.6913071, 23.3065090
3: -13.4087973, 19.2263279, -14.0091572, 20.0458050, -33.4546013, 33.2354851
4: -9.0766134, 18.7372856, -9.4518366, 19.5683060, -28.6449184, 28.1891212

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065092, upper bound: 27.2079594
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2022087, upper bound: 27.1991391
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2086358, upper bound: 27.2087559
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3950882, 11.2541189, -4.6073575, 11.7263937, -16.1214809, 15.8614759
1: -11.9085436, 17.1819000, -12.3935480, 17.7883129, -29.6968555, 29.5754471
2: -7.6134439, 15.4435234, -7.8905721, 16.0862961, -23.6997395, 23.3340950
3: -13.4186106, 19.2597256, -14.0091572, 20.0458050, -33.4644165, 33.2688828
4: -9.0927963, 18.7687836, -9.4518366, 19.5683060, -28.6611023, 28.2206192

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065092, upper bound: 27.2079594
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2022087, upper bound: 27.1994345
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2086358, upper bound: 27.2090513
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3901463, 11.2314816, -4.6157975, 11.7587185, -16.1488647, 15.8472786
1: -11.9016247, 17.1665440, -12.4065723, 17.8174953, -29.7191200, 29.5731144
2: -7.6050115, 15.4159374, -7.9025259, 16.1275082, -23.7325191, 23.3184624
3: -13.4087973, 19.2263279, -14.0258732, 20.0947495, -33.5035477, 33.2522011
4: -9.0766134, 18.7372856, -9.4722614, 19.6183720, -28.6949825, 28.2095470

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065092, upper bound: 27.2077200
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3950882, 11.2541189, -4.6157975, 11.7587185, -16.1538067, 15.8699169
1: -11.9085436, 17.1819000, -12.4065723, 17.8174953, -29.7260399, 29.5884724
2: -7.6134439, 15.4435234, -7.9025259, 16.1275082, -23.7409515, 23.3460484
3: -13.4186106, 19.2597256, -14.0258732, 20.0947495, -33.5133591, 33.2855988
4: -9.0927963, 18.7687836, -9.4722614, 19.6183720, -28.7111683, 28.2410450

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065092, upper bound: 27.2077200
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.5331025, 11.6178961, -16.1319008, 16.0201263
1: -12.1504660, 17.4375305, -12.2875996, 17.7315407, -29.8820038, 29.7251301
2: -7.7290149, 15.7812500, -7.8551126, 15.9343672, -23.6633816, 23.6363621
3: -13.7328539, 19.6459293, -13.8514175, 19.8753357, -33.6081886, 33.4973373
4: -9.2536077, 19.1997623, -9.3789968, 19.3634605, -28.6170692, 28.5787582

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081361, upper bound: 27.2089634
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081361, upper bound: 27.2090474
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.5331025, 11.6178961, -16.2361832, 16.2985973
1: -12.4215508, 17.8367310, -12.2875996, 17.7315407, -30.1530895, 30.1243305
2: -7.9123993, 16.1408329, -7.8551126, 15.9343672, -23.8467636, 23.9959450
3: -14.0420866, 20.1092472, -13.8514175, 19.8753357, -33.9174232, 33.9606628
4: -9.4768257, 19.6367245, -9.3789968, 19.3634605, -28.8402863, 29.0157204

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2083314, upper bound: 27.2089634
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2083314, upper bound: 27.2090474
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.5361848, 11.6357536, -16.1497574, 16.0232086
1: -12.1504660, 17.4375305, -12.2895918, 17.7398758, -29.8903408, 29.7271233
2: -7.7290149, 15.7812500, -7.8600860, 15.9560280, -23.6850433, 23.6413364
3: -13.7328539, 19.6459293, -13.8556681, 19.9008865, -33.6337395, 33.5015869
4: -9.2536077, 19.1997623, -9.3909245, 19.3876572, -28.6412659, 28.5906868

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2064352
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2064352
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.5361848, 11.6357536, -16.2540417, 16.3016758
1: -12.4215508, 17.8367310, -12.2895918, 17.7398758, -30.1614265, 30.1263180
2: -7.9123993, 16.1408329, -7.8600860, 15.9560280, -23.8684254, 24.0009193
3: -14.0420866, 20.1092472, -13.8556681, 19.9008865, -33.9429741, 33.9649162
4: -9.4768257, 19.6367245, -9.3909245, 19.3876572, -28.8644829, 29.0276489

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.5218935, 11.5179110, -4.5331025, 11.6178961, -16.1397896, 16.0510120
1: -12.1622829, 17.4638939, -12.2875996, 17.7315407, -29.8938217, 29.7514935
2: -7.7405324, 15.8200169, -7.8551126, 15.9343672, -23.6749001, 23.6751289
3: -13.7481842, 19.6917896, -13.8514175, 19.8753357, -33.6235199, 33.5432053
4: -9.2733250, 19.2468376, -9.3789968, 19.3634605, -28.6367855, 28.6258354

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065132, upper bound: 27.2088935
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065132, upper bound: 27.2089775
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.5892553, 11.7021198, -4.5331025, 11.6178961, -16.2071514, 16.2352219
1: -12.3333263, 17.7274017, -12.2875996, 17.7315407, -30.0648670, 30.0150013
2: -7.8591914, 16.0541019, -7.8551126, 15.9343672, -23.7935562, 23.9092140
3: -13.9426527, 19.9966927, -13.8514175, 19.8753357, -33.8179893, 33.8481064
4: -9.4191265, 19.5313320, -9.3789968, 19.3634605, -28.7825851, 28.9103279

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063492, upper bound: 27.2088935
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2065132, upper bound: 27.2089775
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.5218935, 11.5179110, -4.5361848, 11.6357536, -16.1576462, 16.0540924
1: -12.1622829, 17.4638939, -12.2895918, 17.7398758, -29.9021587, 29.7534847
2: -7.7405324, 15.8200169, -7.8600860, 15.9560280, -23.6965599, 23.6801033
3: -13.7481842, 19.6917896, -13.8556681, 19.9008865, -33.6490707, 33.5474586
4: -9.2733250, 19.2468376, -9.3909245, 19.3876572, -28.6609821, 28.6377602

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2061725, upper bound: 27.2063653
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2061725, upper bound: 27.2063653
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.5892553, 11.7021198, -4.5361848, 11.6357536, -16.2250080, 16.2383022
1: -12.3333263, 17.7274017, -12.2895918, 17.7398758, -30.0732021, 30.0169926
2: -7.8591914, 16.0541019, -7.8600860, 15.9560280, -23.8152199, 23.9141884
3: -13.9426527, 19.9966927, -13.8556681, 19.9008865, -33.8435402, 33.8523560
4: -9.4191265, 19.5313320, -9.3909245, 19.3876572, -28.8067837, 28.9222546

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.6073575, 11.7263937, -16.2403965, 16.0943813
1: -12.1504660, 17.4375305, -12.3935480, 17.7883129, -29.9387779, 29.8310776
2: -7.7290149, 15.7812500, -7.8905721, 16.0862961, -23.8153114, 23.6718216
3: -13.7328539, 19.6459293, -14.0091572, 20.0458050, -33.7786598, 33.6550827
4: -9.2536077, 19.1997623, -9.4518366, 19.5683060, -28.8219147, 28.6515999

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2080055, upper bound: 27.2080055
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2080055, upper bound: 27.2082008
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.6073575, 11.7263937, -16.3446808, 16.3728504
1: -12.4215508, 17.8367310, -12.3935480, 17.7883129, -30.2098618, 30.2302742
2: -7.9123993, 16.1408329, -7.8905721, 16.0862961, -23.9986935, 24.0314045
3: -14.0420866, 20.1092472, -14.0091572, 20.0458050, -34.0878906, 34.1184044
4: -9.4768257, 19.6367245, -9.4518366, 19.5683060, -29.0451317, 29.0885601

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2082008, upper bound: 27.2080055
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2082008, upper bound: 27.2082008
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.6157975, 11.7587185, -16.2727222, 16.1028214
1: -12.1504660, 17.4375305, -12.4065723, 17.8174953, -29.9679604, 29.8441029
2: -7.7290149, 15.7812500, -7.9025259, 16.1275082, -23.8565235, 23.6837769
3: -13.7328539, 19.6459293, -14.0258732, 20.0947495, -33.8276024, 33.6717987
4: -9.2536077, 19.1997623, -9.4722614, 19.6183720, -28.8719788, 28.6720238

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2039279, upper bound: 27.2050223
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.6157975, 11.7587185, -16.3770065, 16.3812904
1: -12.4215508, 17.8367310, -12.4065723, 17.8174953, -30.2390461, 30.2433033
2: -7.9123993, 16.1408329, -7.9025259, 16.1275082, -24.0399055, 24.0433578
3: -14.0420866, 20.1092472, -14.0258732, 20.0947495, -34.1368370, 34.1351204
4: -9.4768257, 19.6367245, -9.4722614, 19.6183720, -29.0951977, 29.1089859

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2044234, upper bound: 27.2052239
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.5260482, 11.5353231, -4.6073575, 11.7263937, -16.2524414, 16.1426811
1: -12.1748428, 17.5010700, -12.3935480, 17.7883129, -29.9631557, 29.8946152
2: -7.7485857, 15.8332081, -7.8905721, 16.0862961, -23.8348808, 23.7237797
3: -13.7613878, 19.7208557, -14.0091572, 20.0458050, -33.8071938, 33.7300110
4: -9.2842464, 19.2679138, -9.4518366, 19.5683060, -28.8525524, 28.7197495

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2026879
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2032311, upper bound: 27.2036167
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7649322, 12.2145834, -4.5843573, 11.6642141, -16.4291420, 16.7989388
1: -12.8142519, 18.5138855, -12.3311338, 17.6961212, -30.5103722, 30.8450165
2: -8.1534843, 16.7714787, -7.8505945, 16.0018139, -24.1552982, 24.6220741
3: -14.4847021, 20.8651981, -13.9378748, 19.9408207, -34.4255219, 34.8030663
4: -9.7736263, 20.4057159, -9.4033470, 19.4660606, -29.2396870, 29.8090611

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2027226, upper bound: 27.2029033
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2030770, upper bound: 27.2029033
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.5260482, 11.5353231, -4.6157975, 11.7587185, -16.2847652, 16.1511211
1: -12.1748428, 17.5010700, -12.4065723, 17.8174953, -29.9923382, 29.9076424
2: -7.7485857, 15.8332081, -7.9025259, 16.1275082, -23.8760948, 23.7357330
3: -13.7613878, 19.7208557, -14.0258732, 20.0947495, -33.8561363, 33.7467270
4: -9.2842464, 19.2679138, -9.4722614, 19.6183720, -28.9026184, 28.7401752

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2040526, upper bound: 27.2041884
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2040526, upper bound: 27.2041884
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7649322, 12.2145834, -4.5932736, 11.6978598, -16.4627857, 16.8078575
1: -12.8142519, 18.5138855, -12.3452425, 17.7272835, -30.5415325, 30.8591213
2: -8.1534843, 16.7714787, -7.8632298, 16.0448189, -24.1983032, 24.6347084
3: -14.4847021, 20.8651981, -13.9557943, 19.9920387, -34.4767380, 34.8209801
4: -9.7736263, 20.4057159, -9.4246941, 19.5182667, -29.2918930, 29.8304100

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031183
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031811
time: 2.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.82 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1966358
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1966358
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2067467
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2067750
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2070811
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1997698, upper bound: 27.2070811
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1966358, upper bound: 27.1997698
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1966358, upper bound: 27.2007912
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2067467, upper bound: 27.2085488
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2067467, upper bound: 27.2095701
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1906803, upper bound: 27.1906803
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.1906803, upper bound: 27.2007912
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2008195, upper bound: 27.1991030
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2008195, upper bound: 27.2095984
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2022087, upper bound: 27.1991391
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2086358, upper bound: 27.2087559
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2022087, upper bound: 27.1994345
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2086358, upper bound: 27.2090513
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2065092, upper bound: 27.2077200
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2065092, upper bound: 27.2077200
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2081361, upper bound: 27.2089634
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2081361, upper bound: 27.2090474
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2083314, upper bound: 27.2089634
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2083314, upper bound: 27.2090474
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2064352
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2079594, upper bound: 27.2064352
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2081547, upper bound: 27.2064352
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2065132, upper bound: 27.2088935
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2065132, upper bound: 27.2089775
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2063492, upper bound: 27.2088935
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2065132, upper bound: 27.2089775
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2061725, upper bound: 27.2063653
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2061725, upper bound: 27.2063653
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2063365, upper bound: 27.2063653
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2080055, upper bound: 27.2080055
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2080055, upper bound: 27.2082008
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2082008, upper bound: 27.2080055
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2082008, upper bound: 27.2082008
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2039279, upper bound: 27.2050223
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2044234, upper bound: 27.2052239
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2026879
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2032311, upper bound: 27.2036167
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2027226, upper bound: 27.2029033
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2030770, upper bound: 27.2029033
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2040526, upper bound: 27.2041884
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2040526, upper bound: 27.2041884
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031183
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031811

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.9416430, 10.0241356, -3.9416430, 10.0241356, -13.9657784, 13.9657784
1: -10.6106949, 15.3450384, -10.6106949, 15.3450384, -25.9557343, 25.9557343
2: -6.7955890, 13.7306890, -6.7955890, 13.7306890, -20.5262775, 20.5262775
3: -11.9359779, 17.1758213, -11.9359779, 17.1758213, -29.1117954, 29.1117954
4: -8.1233921, 16.6995983, -8.1233921, 16.6995983, -24.8229904, 24.8229904

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1962657
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1995024, upper bound: 27.1966358
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.9416430, 10.0241356, -4.1428089, 10.6177177, -14.5593605, 14.1669445
1: -10.6106949, 15.3450384, -11.2123156, 16.2294140, -26.8401070, 26.5573540
2: -6.7955890, 13.7306890, -7.1643281, 14.5773115, -21.3729000, 20.8950176
3: -11.9359779, 17.1758213, -12.6359673, 18.1798058, -30.1157799, 29.8117886
4: -8.1233921, 16.6995983, -8.5602360, 17.7325573, -25.8559494, 25.2598343

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1962657
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1995024, upper bound: 27.1966358
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.1433077, 10.5423946, -3.9416430, 10.0241356, -14.1674433, 14.4840374
1: -11.1630354, 16.1227474, -10.6106949, 15.3450384, -26.5080700, 26.7334404
2: -7.1534371, 14.4338942, -6.7955890, 13.7306890, -20.8841248, 21.2294827
3: -12.5585098, 18.0531578, -11.9359779, 17.1758213, -29.7343311, 29.9891300
4: -8.5494747, 17.5393734, -8.1233921, 16.6995983, -25.2490730, 25.6627636

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2007912, upper bound: 27.1970684
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2001296, upper bound: 27.1970683
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.1433077, 10.5423946, -4.1428089, 10.6177177, -14.7610254, 14.6852036
1: -11.1630354, 16.1227474, -11.2123156, 16.2294140, -27.3924408, 27.3350582
2: -7.1534371, 14.4338942, -7.1643281, 14.5773115, -21.7307415, 21.5982227
3: -12.5585098, 18.0531578, -12.6359673, 18.1798058, -30.7383156, 30.6891251
4: -8.5494747, 17.5393734, -8.5602360, 17.7325573, -26.2820320, 26.0996075

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2001296, upper bound: 27.1970683
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.9416430, 10.0241356, -4.1433077, 10.5423946, -14.4840374, 14.1674433
1: -10.6106949, 15.3450384, -11.1630354, 16.1227474, -26.7334385, 26.5080681
2: -6.7955890, 13.7306890, -7.1534371, 14.4338942, -21.2294827, 20.8841248
3: -11.9359779, 17.1758213, -12.5585098, 18.0531578, -29.9891300, 29.7343311
4: -8.1233921, 16.6995983, -8.5494747, 17.5393734, -25.6627636, 25.2490730

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2061579, upper bound: 27.2057151
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2060852
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9416430, 10.0241356, -4.3870020, 11.2383909, -15.1800337, 14.4111357
1: -10.6106949, 15.3450384, -11.8893881, 17.1601810, -27.7708759, 27.2344227
2: -6.7955890, 13.7306890, -7.6013036, 15.4227715, -22.2183609, 21.3319912
3: -11.9359779, 17.1758213, -13.3972139, 19.2366486, -31.1726227, 30.5730343
4: -8.1233921, 16.6995983, -9.0787373, 18.7438374, -26.8672295, 25.7783356

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2061579, upper bound: 27.2058551
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2062252
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.1433077, 10.5423946, -4.1433077, 10.5423946, -14.6857023, 14.6857023
1: -11.1630354, 16.1227474, -11.1630354, 16.1227474, -27.2857742, 27.2857723
2: -7.1534371, 14.4338942, -7.1534371, 14.4338942, -21.5873280, 21.5873280
3: -12.5585098, 18.0531578, -12.5585098, 18.0531578, -30.6116676, 30.6116676
4: -8.5494747, 17.5393734, -8.5494747, 17.5393734, -26.0888481, 26.0888481

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2071792, upper bound: 27.2065178
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2064862, upper bound: 27.2065177
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1433077, 10.5423946, -4.3870020, 11.2383909, -15.3816986, 14.9293957
1: -11.1630354, 16.1227474, -11.8893881, 17.1601810, -28.3232098, 28.0121288
2: -7.1534371, 14.4338942, -7.6013036, 15.4227715, -22.5762024, 22.0351963
3: -12.5585098, 18.0531578, -13.3972139, 19.2366486, -31.7951584, 31.4503708
4: -8.5494747, 17.5393734, -9.0787373, 18.7438374, -27.2933121, 26.6181087

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2071792, upper bound: 27.2065869
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2064862, upper bound: 27.2065476
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.1428089, 10.6177177, -3.9416430, 10.0241356, -14.1669445, 14.5593605
1: -11.2123156, 16.2294140, -10.6106949, 15.3450384, -26.5573540, 26.8401089
2: -7.1643281, 14.5773115, -6.7955890, 13.7306890, -20.8950176, 21.3729000
3: -12.6359673, 18.1798058, -11.9359779, 17.1758213, -29.8117886, 30.1157799
4: -8.5602360, 17.7325573, -8.1233921, 16.6995983, -25.2598343, 25.8559494

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1663970, upper bound: 27.1791007
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966358, upper bound: 27.1997495
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.1428089, 10.6177177, -4.1433077, 10.5423946, -14.6852036, 14.7610254
1: -11.2123156, 16.2294140, -11.1630354, 16.1227474, -27.3350563, 27.3924408
2: -7.1643281, 14.5773115, -7.1534371, 14.4338942, -21.5982227, 21.7307415
3: -12.6359673, 18.1798058, -12.5585098, 18.0531578, -30.6891251, 30.7383156
4: -8.5602360, 17.7325573, -8.5494747, 17.5393734, -26.0996094, 26.2820320

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1663970, upper bound: 27.1817158
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966358, upper bound: 27.2007646
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.3870020, 11.2383909, -3.9416430, 10.0241356, -14.4111366, 15.1800337
1: -11.8893881, 17.1601810, -10.6106949, 15.3450384, -27.2344246, 27.7708759
2: -7.6013036, 15.4227715, -6.7955890, 13.7306890, -21.3319893, 22.2183590
3: -13.3972139, 19.2366486, -11.9359779, 17.1758213, -30.5730362, 31.1726227
4: -9.0787373, 18.7438374, -8.1233921, 16.6995983, -25.7783356, 26.8672295

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2067750, upper bound: 27.2078943
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966358, upper bound: 27.2082814
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.3870020, 11.2383909, -4.1433077, 10.5423946, -14.9293957, 15.3816986
1: -11.8893881, 17.1601810, -11.1630354, 16.1227474, -28.0121307, 28.3232098
2: -7.6013036, 15.4227715, -7.1534371, 14.4338942, -22.0351963, 22.5762024
3: -13.3972139, 19.2366486, -12.5585098, 18.0531578, -31.4503670, 31.7951584
4: -9.0787373, 18.7438374, -8.5494747, 17.5393734, -26.6181087, 27.2933121

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2067750, upper bound: 27.2085214
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2062252, upper bound: 27.2089086
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.1428089, 10.6177177, -4.3870020, 11.2383909, -15.3811998, 15.0047178
1: -11.2123156, 16.2294140, -11.8893881, 17.1601810, -28.3724937, 28.1187992
2: -7.1643281, 14.5773115, -7.6013036, 15.4227715, -22.5870991, 22.1786098
3: -12.6359673, 18.1798058, -13.3972139, 19.2366486, -31.8726158, 31.5770187
4: -8.5602360, 17.7325573, -9.0787373, 18.7438374, -27.3040733, 26.8112946

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1651217, upper bound: 27.1817158
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1906600, upper bound: 27.2007646
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.3870020, 11.2383909, -4.1428089, 10.6177177, -15.0047188, 15.3811998
1: -11.8893881, 17.1601810, -11.2123156, 16.2294140, -28.1187973, 28.3724937
2: -7.6013036, 15.4227715, -7.1643281, 14.5773115, -22.1786098, 22.5870991
3: -13.3972139, 19.2366486, -12.6359673, 18.1798058, -31.5770187, 31.8726158
4: -9.0787373, 18.7438374, -8.5602360, 17.7325573, -26.8112946, 27.3040733

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2008195, upper bound: 27.1987273
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1906803, upper bound: 27.1991030
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3870020, 11.2383909, -4.3870020, 11.2383909, -15.6253881, 15.6253891
1: -11.8893881, 17.1601810, -11.8893881, 17.1601810, -29.0495663, 29.0495644
2: -7.6013036, 15.4227715, -7.6013036, 15.4227715, -23.0240707, 23.0240688
3: -13.3972139, 19.2366486, -13.3972139, 19.2366486, -32.6338615, 32.6338615
4: -9.0787373, 18.7438374, -9.0787373, 18.7438374, -27.8225746, 27.8225746

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2008195, upper bound: 27.2086614
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2002696, upper bound: 27.2090465
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.3884568, 11.2270498, -4.8190107, 12.2475748, -16.6360264, 16.0460567
1: -11.8970470, 17.1600285, -12.9562016, 18.5419273, -30.4389725, 30.1162300
2: -7.6020870, 15.4099369, -8.2493114, 16.7840824, -24.3861694, 23.6592484
3: -13.4035540, 19.2188244, -14.6557646, 20.9277935, -34.3313446, 33.8745880
4: -9.0730972, 18.7300339, -9.8816185, 20.4151955, -29.4882927, 28.6116524

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.3901463, 11.2314816, -4.6069040, 11.7251883, -16.1153336, 15.8383856
1: -11.9016247, 17.1665440, -12.3923378, 17.7865410, -29.6881657, 29.5588779
2: -7.6050115, 15.4159374, -7.8898001, 16.0846634, -23.6896725, 23.3057365
3: -13.4087973, 19.2263279, -14.0077677, 20.0437603, -33.4525566, 33.2340965
4: -9.0766134, 18.7372856, -9.4508972, 19.5663261, -28.6429348, 28.1881828

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.3933611, 11.2496071, -4.8190107, 12.2475748, -16.6409340, 16.0686150
1: -11.9038792, 17.1752815, -12.9562016, 18.5419273, -30.4458027, 30.1314831
2: -7.6104593, 15.4374170, -8.2493114, 16.7840824, -24.3945408, 23.6867294
3: -13.4132690, 19.2520943, -14.6557646, 20.9277935, -34.3410568, 33.9078560
4: -9.0892067, 18.7614040, -9.8816185, 20.4151955, -29.5043964, 28.6430206

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.3950882, 11.2541189, -4.6069040, 11.7251883, -16.1202774, 15.8610229
1: -11.9085436, 17.1819000, -12.3923378, 17.7865410, -29.6950836, 29.5742378
2: -7.6134439, 15.4435234, -7.8898001, 16.0846634, -23.6981068, 23.3333206
3: -13.4186106, 19.2597256, -14.0077677, 20.0437603, -33.4623718, 33.2674942
4: -9.0927963, 18.7687836, -9.4508972, 19.5663261, -28.6591225, 28.2196808

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.3901463, 11.2314816, -4.5218935, 11.5179110, -15.9080563, 15.7533751
1: -11.9016247, 17.1665440, -12.1622829, 17.4638939, -29.3655186, 29.3288212
2: -7.6050115, 15.4159374, -7.7405324, 15.8200169, -23.4250278, 23.1564693
3: -13.4087973, 19.2263279, -13.7481842, 19.6917896, -33.1005859, 32.9745102
4: -9.0766134, 18.7372856, -9.2733250, 19.2468376, -28.3234482, 28.0106106

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2088935, upper bound: 27.2065132
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2088689, upper bound: 27.2065132
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.3901463, 11.2314816, -4.5892553, 11.7021198, -16.0922642, 15.8207369
1: -11.9016247, 17.1665440, -12.3333263, 17.7274017, -29.6290264, 29.4998665
2: -7.6050115, 15.4159374, -7.8591914, 16.0541019, -23.6591129, 23.2751274
3: -13.4087973, 19.2263279, -13.9426527, 19.9966927, -33.4054909, 33.1689796
4: -9.0766134, 18.7372856, -9.4191265, 19.5313320, -28.6079407, 28.1564121

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2088935, upper bound: 27.2065132
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2088935, upper bound: 27.2065132
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.3950882, 11.2541189, -4.5218935, 11.5179110, -15.9129982, 15.7760124
1: -11.9085436, 17.1819000, -12.1622829, 17.4638939, -29.3724365, 29.3441830
2: -7.6134439, 15.4435234, -7.7405324, 15.8200169, -23.4334602, 23.1840553
3: -13.4186106, 19.2597256, -13.7481842, 19.6917896, -33.1104012, 33.0079117
4: -9.0927963, 18.7687836, -9.2733250, 19.2468376, -28.3396339, 28.0421085

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1940623, upper bound: 27.2012852
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2053661, upper bound: 27.2073604
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3950882, 11.2541189, -4.5892553, 11.7021198, -16.0972061, 15.8433743
1: -11.9085436, 17.1819000, -12.3333263, 17.7274017, -29.6359444, 29.5152264
2: -7.6134439, 15.4435234, -7.8591914, 16.0541019, -23.6675453, 23.3027115
3: -13.4186106, 19.2597256, -13.9426527, 19.9966927, -33.4152946, 33.2023773
4: -9.0927963, 18.7687836, -9.4191265, 19.5313320, -28.6241283, 28.1879082

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.4417982, 11.3863373, -15.9003401, 15.9288187
1: -12.1504660, 17.4375305, -12.0495319, 17.3939400, -29.5444050, 29.4870625
2: -7.7290149, 15.7812500, -7.6975031, 15.6358109, -23.3648243, 23.4787502
3: -13.7328539, 19.6459293, -13.5812998, 19.4890900, -33.2219429, 33.2272186
4: -9.2536077, 19.1997623, -9.1859016, 19.0023212, -28.2559280, 28.3856640

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.5655098, 11.7029524, -16.2169552, 16.0525341
1: -12.1504660, 17.4375305, -12.3700094, 17.8537140, -30.0041809, 29.8075409
2: -7.7290149, 15.7812500, -7.9141040, 16.0532703, -23.7822838, 23.6953545
3: -13.7328539, 19.6459293, -13.9453306, 20.0201893, -33.7530441, 33.5912514
4: -9.2536077, 19.1997623, -9.4492016, 19.5061398, -28.7597466, 28.6489620

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.4417982, 11.3863373, -16.0046234, 16.2072868
1: -12.4215508, 17.8367310, -12.0495319, 17.3939400, -29.8154888, 29.8862591
2: -7.9123993, 16.1408329, -7.6975031, 15.6358109, -23.5482044, 23.8383369
3: -14.0420866, 20.1092472, -13.5812998, 19.4890900, -33.5311775, 33.6905441
4: -9.4768257, 19.6367245, -9.1859016, 19.0023212, -28.4791470, 28.8226261

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1987789, upper bound: 27.2022074
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2082960, upper bound: 27.2089442
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.5655098, 11.7029524, -16.3212376, 16.3310013
1: -12.4215508, 17.8367310, -12.3700094, 17.8537140, -30.2752647, 30.2067413
2: -7.9123993, 16.1408329, -7.9141040, 16.0532703, -23.9656658, 24.0549374
3: -14.0420866, 20.1092472, -13.9453306, 20.0201893, -34.0622749, 34.0545769
4: -9.4768257, 19.6367245, -9.4492016, 19.5061398, -28.9829655, 29.0859222

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1987789, upper bound: 27.2022646
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1987789, upper bound: 27.2089442
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.4454331, 11.4058523, -15.9198551, 15.9324570
1: -12.1504660, 17.4375305, -12.0527382, 17.4043102, -29.5547733, 29.4902687
2: -7.7290149, 15.7812500, -7.7037153, 15.6596470, -23.3886623, 23.4849606
3: -13.7328539, 19.6459293, -13.5869904, 19.5171146, -33.2499695, 33.2329063
4: -9.2536077, 19.1997623, -9.1995935, 19.0287514, -28.2823601, 28.3993568

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.5239525, 11.6063519, -16.1203556, 16.0109749
1: -12.1504660, 17.4375305, -12.2514620, 17.6956997, -29.8461628, 29.6889915
2: -7.7290149, 15.7812500, -7.8417292, 15.9208469, -23.6498604, 23.6229782
3: -13.7328539, 19.6459293, -13.8126240, 19.8538227, -33.5866776, 33.4585419
4: -9.2536077, 19.1997623, -9.3680286, 19.3439426, -28.5975494, 28.5677910

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.4454331, 11.4058523, -16.0241394, 16.2109261
1: -12.4215508, 17.8367310, -12.0527382, 17.4043102, -29.8258591, 29.8894691
2: -7.9123993, 16.1408329, -7.7037153, 15.6596470, -23.5720463, 23.8445435
3: -14.0420866, 20.1092472, -13.5869904, 19.5171146, -33.5592003, 33.6962318
4: -9.4768257, 19.6367245, -9.1995935, 19.0287514, -28.5055771, 28.8363190

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1986035, upper bound: 27.1998167
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081207, upper bound: 27.2064161
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.5239525, 11.6063519, -16.2246399, 16.2894440
1: -12.4215508, 17.8367310, -12.2514620, 17.6956997, -30.1172485, 30.0881901
2: -7.9123993, 16.1408329, -7.8417292, 15.9208469, -23.8332424, 23.9825630
3: -14.0420866, 20.1092472, -13.8126240, 19.8538227, -33.8959084, 33.9218712
4: -9.4768257, 19.6367245, -9.3680286, 19.3439426, -28.8207684, 29.0047531

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1986035, upper bound: 27.1998167
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2081207, upper bound: 27.2064161
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.5218935, 11.5179110, -4.4417982, 11.3863373, -15.9082308, 15.9597006
1: -12.1622829, 17.4638939, -12.0495319, 17.3939400, -29.5562210, 29.5134258
2: -7.7405324, 15.8200169, -7.6975031, 15.6358109, -23.3763428, 23.5175171
3: -13.7481842, 19.6917896, -13.5812998, 19.4890900, -33.2372742, 33.2730904
4: -9.2733250, 19.2468376, -9.1859016, 19.0023212, -28.2756462, 28.4327393

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1959394, upper bound: 27.2058958
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2075371, upper bound: 27.2094537
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.5218935, 11.5179110, -4.5655098, 11.7029524, -16.2248459, 16.0834179
1: -12.1622829, 17.4638939, -12.3700094, 17.8537140, -30.0159969, 29.8339043
2: -7.7405324, 15.8200169, -7.9141040, 16.0532703, -23.7938023, 23.7341194
3: -13.7481842, 19.6917896, -13.9453306, 20.0201893, -33.7683716, 33.6371193
4: -9.2733250, 19.2468376, -9.4492016, 19.5061398, -28.7794647, 28.6960354

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1935186, upper bound: 27.2059385
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1959394, upper bound: 27.2095082
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.5892553, 11.7021198, -4.4417982, 11.3863373, -15.9755907, 16.1439133
1: -12.3333263, 17.7274017, -12.0495319, 17.3939400, -29.7272644, 29.7769337
2: -7.8591914, 16.0541019, -7.6975031, 15.6358109, -23.4949970, 23.7516041
3: -13.9426527, 19.9966927, -13.5812998, 19.4890900, -33.4317436, 33.5779839
4: -9.4191265, 19.5313320, -9.1859016, 19.0023212, -28.4214458, 28.7172337

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966853, upper bound: 27.2050286
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966853, upper bound: 27.2088638
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.5892553, 11.7021198, -4.5655098, 11.7029524, -16.2922039, 16.2676296
1: -12.3333263, 17.7274017, -12.3700094, 17.8537140, -30.1870403, 30.0974121
2: -7.8591914, 16.0541019, -7.9141040, 16.0532703, -23.9124584, 23.9682045
3: -13.9426527, 19.9966927, -13.9453306, 20.0201893, -33.9628410, 33.9420166
4: -9.4191265, 19.5313320, -9.4492016, 19.5061398, -28.9252663, 28.9805298

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966854, upper bound: 27.2050286
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1966854, upper bound: 27.2088638
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.5218935, 11.5179110, -4.4454331, 11.4058523, -15.9277458, 15.9633427
1: -12.1622829, 17.4638939, -12.0527382, 17.4043102, -29.5665913, 29.5166321
2: -7.7405324, 15.8200169, -7.7037153, 15.6596470, -23.4001789, 23.5237274
3: -13.7481842, 19.6917896, -13.5869904, 19.5171146, -33.2652969, 33.2787781
4: -9.2733250, 19.2468376, -9.1995935, 19.0287514, -28.3020763, 28.4464302

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1957628, upper bound: 27.2027216
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2073604, upper bound: 27.2058121
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.5218935, 11.5179110, -4.5239525, 11.6063519, -16.1282463, 16.0418587
1: -12.1622829, 17.4638939, -12.2514620, 17.6956997, -29.8579807, 29.7153549
2: -7.7405324, 15.8200169, -7.8417292, 15.9208469, -23.6613789, 23.6617451
3: -13.7481842, 19.6917896, -13.8126240, 19.8538227, -33.6020050, 33.5044136
4: -9.2733250, 19.2468376, -9.3680286, 19.3439426, -28.6172676, 28.6148663

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1957628, upper bound: 27.2027216
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1957628, upper bound: 27.2058121
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.5892553, 11.7021198, -4.4454331, 11.4058523, -15.9951057, 16.1475525
1: -12.3333263, 17.7274017, -12.0527382, 17.4043102, -29.7376366, 29.7801399
2: -7.8591914, 16.0541019, -7.7037153, 15.6596470, -23.5188389, 23.7578106
3: -13.9426527, 19.9966927, -13.5869904, 19.5171146, -33.4597664, 33.5836716
4: -9.4191265, 19.5313320, -9.1995935, 19.0287514, -28.4478779, 28.7309265

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1965087, upper bound: 27.2018912
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2048693, upper bound: 27.2052222
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.5892553, 11.7021198, -4.5239525, 11.6063519, -16.1956062, 16.2260704
1: -12.3333263, 17.7274017, -12.2514620, 17.6956997, -30.0290260, 29.9788628
2: -7.8591914, 16.0541019, -7.8417292, 15.9208469, -23.7800350, 23.8958302
3: -13.9426527, 19.9966927, -13.8126240, 19.8538227, -33.7964745, 33.8093109
4: -9.4191265, 19.5313320, -9.3680286, 19.3439426, -28.7630672, 28.8993607

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1965087, upper bound: 27.2018912
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2048693, upper bound: 27.2052222
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.5140038, 11.4870243, -16.0010281, 16.0010281
1: -12.1504660, 17.4375305, -12.1504660, 17.4375305, -29.5879974, 29.5879974
2: -7.7290149, 15.7812500, -7.7290149, 15.7812500, -23.5102654, 23.5102654
3: -13.7328539, 19.6459293, -13.7328539, 19.6459293, -33.3787766, 33.3787766
4: -9.2536077, 19.1997623, -9.2536077, 19.1997623, -28.4533691, 28.4533691

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026878, upper bound: 27.2026879
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2026879
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.6182880, 11.7654943, -16.2794952, 16.1053123
1: -12.1504660, 17.4375305, -12.4215508, 17.8367310, -29.9871941, 29.8590813
2: -7.7290149, 15.7812500, -7.9123993, 16.1408329, -23.8698483, 23.6936474
3: -13.7328539, 19.6459293, -14.0420866, 20.1092472, -33.8421021, 33.6880112
4: -9.2536077, 19.1997623, -9.4768257, 19.6367245, -28.8903313, 28.6765881

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2034954
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2031834
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.5140038, 11.4870243, -16.1053123, 16.2794952
1: -12.4215508, 17.8367310, -12.1504660, 17.4375305, -29.8590813, 29.9871922
2: -7.9123993, 16.1408329, -7.7290149, 15.7812500, -23.6936474, 23.8698483
3: -14.0420866, 20.1092472, -13.7328539, 19.6459293, -33.6880112, 33.8421021
4: -9.4768257, 19.6367245, -9.2536077, 19.1997623, -28.6765881, 28.8903313

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2028204, upper bound: 27.2026879
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2028338, upper bound: 27.2027958
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.6182880, 11.7654943, -16.3837795, 16.3837814
1: -12.4215508, 17.8367310, -12.4215508, 17.8367310, -30.2582779, 30.2582798
2: -7.9123993, 16.1408329, -7.9123993, 16.1408329, -24.0532303, 24.0532322
3: -14.0420866, 20.1092472, -14.0420866, 20.1092472, -34.1513329, 34.1513329
4: -9.4768257, 19.6367245, -9.4768257, 19.6367245, -29.1135502, 29.1135502

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2028203, upper bound: 27.2031507
time: 1.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2028338, upper bound: 27.2031031
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.5140038, 11.4870243, -4.5260482, 11.5353231, -16.0493279, 16.0130730
1: -12.1504660, 17.4375305, -12.1748428, 17.5010700, -29.6515350, 29.6123734
2: -7.7290149, 15.7812500, -7.7485857, 15.8332081, -23.5622234, 23.5298347
3: -13.7328539, 19.6459293, -13.7613878, 19.7208557, -33.4537086, 33.4073143
4: -9.2536077, 19.1997623, -9.2842464, 19.2679138, -28.5215225, 28.4840088

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.4909239, 11.4247599, -4.7649322, 12.2145834, -16.7055073, 16.1896915
1: -12.0876226, 17.3453865, -12.8142519, 18.5138855, -30.6015034, 30.1596355
2: -7.6887441, 15.6966496, -8.1534843, 16.7714787, -24.4602222, 23.8501339
3: -13.6610680, 19.5408993, -14.4847021, 20.8651981, -34.5262642, 34.0256004
4: -9.2048931, 19.0973473, -9.7736263, 20.4057159, -29.6106091, 28.8709736

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.6182880, 11.7654943, -4.5260482, 11.5353231, -16.1536102, 16.2915421
1: -12.4215508, 17.8367310, -12.1748428, 17.5010700, -29.9226189, 30.0115738
2: -7.9123993, 16.1408329, -7.7485857, 15.8332081, -23.7456074, 23.8894196
3: -14.0420866, 20.1092472, -13.7613878, 19.7208557, -33.7629433, 33.8706360
4: -9.4768257, 19.6367245, -9.2842464, 19.2679138, -28.7447395, 28.9209709

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.5948772, 11.7024612, -4.7649322, 12.2145834, -16.8094578, 16.4673920
1: -12.3582277, 17.7435818, -12.8142519, 18.5138855, -30.8721123, 30.5578327
2: -7.8717699, 16.0555305, -8.1534843, 16.7714787, -24.6432495, 24.2090130
3: -13.9697571, 20.0029640, -14.4847021, 20.8651981, -34.8349457, 34.4876671
4: -9.4275064, 19.5335598, -9.7736263, 20.4057159, -29.8332214, 29.3071861

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.5260482, 11.5353231, -4.5140038, 11.4870243, -16.0130711, 16.0493279
1: -12.1748428, 17.5010700, -12.1504660, 17.4375305, -29.6123734, 29.6515350
2: -7.7485857, 15.8332081, -7.7290149, 15.7812500, -23.5298347, 23.5622234
3: -13.7613878, 19.7208557, -13.7328539, 19.6459293, -33.4073105, 33.4537086
4: -9.2842464, 19.2679138, -9.2536077, 19.1997623, -28.4840088, 28.5215225

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2026879
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2026878, upper bound: 27.2026877
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.5260482, 11.5353231, -4.6182880, 11.7654943, -16.2915421, 16.1536102
1: -12.1748428, 17.5010700, -12.4215508, 17.8367310, -30.0115738, 29.9226189
2: -7.7485857, 15.8332081, -7.9123993, 16.1408329, -23.8894196, 23.7456074
3: -13.7613878, 19.7208557, -14.0420866, 20.1092472, -33.8706360, 33.7629433
4: -9.2842464, 19.2679138, -9.4768257, 19.6367245, -28.9209709, 28.7447395

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2032311, upper bound: 27.2036167
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2032311, upper bound: 27.2036167
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.7649322, 12.2145834, -4.4909239, 11.4247599, -16.1896896, 16.7055073
1: -12.8142519, 18.5138855, -12.0876226, 17.3453865, -30.1596355, 30.6015034
2: -8.1534843, 16.7714787, -7.6887441, 15.6966496, -23.8501339, 24.4602222
3: -14.4847021, 20.8651981, -13.6610680, 19.5408993, -34.0256004, 34.5262642
4: -9.7736263, 20.4057159, -9.2048931, 19.0973473, -28.8709736, 29.6106091

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2027226, upper bound: 27.2029033
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2027225, upper bound: 27.2029033
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7649322, 12.2145834, -4.5948772, 11.7024612, -16.4673920, 16.8094578
1: -12.8142519, 18.5138855, -12.3582277, 17.7435818, -30.5578327, 30.8721123
2: -8.1534843, 16.7714787, -7.8717699, 16.0555305, -24.2090130, 24.6432495
3: -14.4847021, 20.8651981, -13.9697571, 20.0029640, -34.4876671, 34.8349457
4: -9.7736263, 20.4057159, -9.4275064, 19.5335598, -29.3071861, 29.8332214

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2030402, upper bound: 27.2034632
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2030402, upper bound: 27.2034632
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.5260482, 11.5353231, -4.5260482, 11.5353231, -16.0613708, 16.0613708
1: -12.1748428, 17.5010700, -12.1748428, 17.5010700, -29.6759129, 29.6759129
2: -7.7485857, 15.8332081, -7.7485857, 15.8332081, -23.5817947, 23.5817947
3: -13.7613878, 19.7208557, -13.7613878, 19.7208557, -33.4822426, 33.4822426
4: -9.2842464, 19.2679138, -9.2842464, 19.2679138, -28.5521603, 28.5521603

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2038437, upper bound: 27.2036632
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2032365, upper bound: 27.2032643
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.5260482, 11.5353231, -4.7649322, 12.2145834, -16.7406311, 16.3002548
1: -12.1748428, 17.5010700, -12.8142519, 18.5138855, -30.6887283, 30.3153210
2: -7.7485857, 15.8332081, -8.1534843, 16.7714787, -24.5200653, 23.9866924
3: -13.7613878, 19.7208557, -14.4847021, 20.8651981, -34.6265831, 34.2055588
4: -9.2842464, 19.2679138, -9.7736263, 20.4057159, -29.6899624, 29.0415401

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2038437, upper bound: 27.2036632
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2032365, upper bound: 27.2032643
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7649322, 12.2145834, -4.4992399, 11.4566574, -16.2215881, 16.7138233
1: -12.8142519, 18.5138855, -12.1005716, 17.3731861, -30.1874390, 30.6144562
2: -8.1534843, 16.7714787, -7.7009625, 15.7368603, -23.8903446, 24.4724407
3: -14.4847021, 20.8651981, -13.6776905, 19.5884762, -34.0731773, 34.5428734
4: -9.7736263, 20.4057159, -9.2254238, 19.1461620, -28.9197884, 29.6311398

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031183
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031183
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7649322, 12.2145834, -4.5660090, 11.6393995, -16.4043312, 16.7805901
1: -12.8142519, 18.5138855, -12.2703667, 17.6345539, -30.4488049, 30.7842503
2: -8.1534843, 16.7714787, -7.8187609, 15.9691486, -24.1226311, 24.5902405
3: -14.4847021, 20.8651981, -13.8707752, 19.8908558, -34.3755569, 34.7359695
4: -9.7736263, 20.4057159, -9.3700552, 19.4285965, -29.2022209, 29.7757721

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031811
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031811
time: 0.94 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.81 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1962657
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1995024, upper bound: 27.1966358
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1962657
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1995024, upper bound: 27.1966358
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2007912, upper bound: 27.1970684
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2001296, upper bound: 27.1970683
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1997698, upper bound: 27.1970684
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2001296, upper bound: 27.1970683
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2061579, upper bound: 27.2057151
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2060852
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2061579, upper bound: 27.2058551
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2062252
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2071792, upper bound: 27.2065178
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2064862, upper bound: 27.2065177
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2071792, upper bound: 27.2065869
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2064862, upper bound: 27.2065476
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1663970, upper bound: 27.1791007
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966358, upper bound: 27.1997495
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1663970, upper bound: 27.1817158
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966358, upper bound: 27.2007646
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2067750, upper bound: 27.2078943
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966358, upper bound: 27.2082814
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2067750, upper bound: 27.2085214
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2062252, upper bound: 27.2089086
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1651217, upper bound: 27.1817158
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1906600, upper bound: 27.2007646
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2008195, upper bound: 27.1987273
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1906803, upper bound: 27.1991030
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2008195, upper bound: 27.2086614
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2002696, upper bound: 27.2090465
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2088935, upper bound: 27.2065132
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2088689, upper bound: 27.2065132
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2088935, upper bound: 27.2065132
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2088935, upper bound: 27.2065132
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1940623, upper bound: 27.2012852
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2053661, upper bound: 27.2073604
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2063653, upper bound: 27.2063365
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1987789, upper bound: 27.2022074
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2082960, upper bound: 27.2089442
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1987789, upper bound: 27.2022646
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1987789, upper bound: 27.2089442
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1986035, upper bound: 27.1998167
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2081207, upper bound: 27.2064161
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1986035, upper bound: 27.1998167
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2081207, upper bound: 27.2064161
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1959394, upper bound: 27.2058958
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2075371, upper bound: 27.2094537
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1935186, upper bound: 27.2059385
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1959394, upper bound: 27.2095082
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966853, upper bound: 27.2050286
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966853, upper bound: 27.2088638
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966854, upper bound: 27.2050286
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1966854, upper bound: 27.2088638
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1957628, upper bound: 27.2027216
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2073604, upper bound: 27.2058121
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1957628, upper bound: 27.2027216
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1957628, upper bound: 27.2058121
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1965087, upper bound: 27.2018912
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2048693, upper bound: 27.2052222
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.1965087, upper bound: 27.2018912
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2048693, upper bound: 27.2052222
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2026878, upper bound: 27.2026879
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2026879
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2034954
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2031834
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2028204, upper bound: 27.2026879
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2028338, upper bound: 27.2027958
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2028203, upper bound: 27.2031507
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2028338, upper bound: 27.2031031
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2029677, upper bound: 27.2029014
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2033572, upper bound: 27.2031030
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2026879, upper bound: 27.2026879
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2026878, upper bound: 27.2026877
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2032311, upper bound: 27.2036167
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2032311, upper bound: 27.2036167
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2027226, upper bound: 27.2029033
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2027225, upper bound: 27.2029033
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2030402, upper bound: 27.2034632
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2030402, upper bound: 27.2034632
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2038437, upper bound: 27.2036632
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2032365, upper bound: 27.2032643
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2038437, upper bound: 27.2036632
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2032365, upper bound: 27.2032643
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031183
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031183
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031811
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 4, lower bound: -27.2031183, upper bound: 27.2031811

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9068065, 9.9074421, -3.9416430, 10.0241356, -13.9309406, 13.8490849
1: -10.5187998, 15.1827583, -10.6106949, 15.3450384, -25.8638382, 25.7934513
2: -6.7327509, 13.5722666, -6.7955890, 13.7306890, -20.4634399, 20.3678551
3: -11.8289604, 16.9825478, -11.9359779, 17.1758213, -29.0047817, 28.9185200
4: -8.0432854, 16.5048981, -8.1233921, 16.6995983, -24.7428837, 24.6282883

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2050879, upper bound: 27.2050879
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2050879, upper bound: 27.2050879
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8896875, 9.8881626, -3.9416430, 10.0241356, -13.9138231, 13.8298054
1: -10.4701576, 15.1473541, -10.6106949, 15.3450384, -25.8151970, 25.7580471
2: -6.7051382, 13.5473413, -6.7955890, 13.7306890, -20.4358273, 20.3429298
3: -11.7754116, 16.9464436, -11.9359779, 17.1758213, -28.9512329, 28.8824177
4: -8.0145988, 16.4772720, -8.1233921, 16.6995983, -24.7141972, 24.6006641

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2050879, upper bound: 27.2054580
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2050879, upper bound: 27.2054580
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9068065, 9.9074421, -4.1428089, 10.6177177, -14.5245237, 14.0502510
1: -10.5187998, 15.1827583, -11.2123156, 16.2294140, -26.7482109, 26.3950691
2: -6.7327509, 13.5722666, -7.1643281, 14.5773115, -21.3100605, 20.7365952
3: -11.8289604, 16.9825478, -12.6359673, 18.1798058, -30.0087662, 29.6185150
4: -8.0432854, 16.5048981, -8.5602360, 17.7325573, -25.7758427, 25.0651321

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791007, upper bound: 27.1634807
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791007, upper bound: 27.1962657
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8896875, 9.8881626, -4.1428089, 10.6177177, -14.5074053, 14.0309715
1: -10.4701576, 15.1473541, -11.2123156, 16.2294140, -26.6995697, 26.3596687
2: -6.7051382, 13.5473413, -7.1643281, 14.5773115, -21.2824497, 20.7116699
3: -11.7754116, 16.9464436, -12.6359673, 18.1798058, -29.9552174, 29.5824108
4: -8.0145988, 16.4772720, -8.5602360, 17.7325573, -25.7471561, 25.0375080

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791007, upper bound: 27.1663970
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1994821, upper bound: 27.1966358
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1111946, 10.4365177, -3.9416430, 10.0241356, -14.1353302, 14.3781605
1: -11.0838718, 15.9800167, -10.6106949, 15.3450384, -26.4289093, 26.5907116
2: -7.0965567, 14.2923832, -6.7955890, 13.7306890, -20.8272457, 21.0879726
3: -12.4648304, 17.8796692, -11.9359779, 17.1758213, -29.6406517, 29.8156433
4: -8.4751015, 17.3669186, -8.1233921, 16.6995983, -25.1746979, 25.4903069

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2057151, upper bound: 27.2058905
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2057151, upper bound: 27.2058905
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0895777, 10.4020500, -3.9416430, 10.0241356, -14.1137123, 14.3436928
1: -11.0181484, 15.9188347, -10.6106949, 15.3450384, -26.3631859, 26.5295296
2: -7.0597200, 14.2448359, -6.7955890, 13.7306890, -20.7904091, 21.0404243
3: -12.3929386, 17.8163891, -11.9359779, 17.1758213, -29.5687599, 29.7523613
4: -8.4366903, 17.3104019, -8.1233921, 16.6995983, -25.1362877, 25.4337940

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2057151, upper bound: 27.2058905
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2057151, upper bound: 27.2058905
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1111946, 10.4365177, -4.1428089, 10.6177177, -14.7289124, 14.5793266
1: -11.0838718, 15.9800167, -11.2123156, 16.2294140, -27.3132820, 27.1923313
2: -7.0965567, 14.2923832, -7.1643281, 14.5773115, -21.6738663, 21.4567108
3: -12.4648304, 17.8796692, -12.6359673, 18.1798058, -30.6446362, 30.5156364
4: -8.4751015, 17.3669186, -8.5602360, 17.7325573, -26.2076569, 25.9271507

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2006416, upper bound: 27.1966667
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1817158, upper bound: 27.1670527
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2007643, upper bound: 27.1970684
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0895777, 10.4020500, -4.1428089, 10.6177177, -14.7072945, 14.5448589
1: -11.0181484, 15.9188347, -11.2123156, 16.2294140, -27.2475624, 27.1311474
2: -7.0597200, 14.2448359, -7.1643281, 14.5773115, -21.6370277, 21.4091644
3: -12.3929386, 17.8163891, -12.6359673, 18.1798058, -30.5727444, 30.4523525
4: -8.4366903, 17.3104019, -8.5602360, 17.7325573, -26.1692467, 25.8706379

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1791007, upper bound: 27.1671489
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2001093, upper bound: 27.1970683
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9068065, 9.9074421, -4.1433077, 10.5423946, -14.4492016, 14.0507498
1: -10.5187998, 15.1827583, -11.1630354, 16.1227474, -26.6415443, 26.3457851
2: -6.7327509, 13.5722666, -7.1534371, 14.4338942, -21.1666451, 20.7256985
3: -11.8289604, 16.9825478, -12.5585098, 18.0531578, -29.8821182, 29.5410576
4: -8.0432854, 16.5048981, -8.5494747, 17.5393734, -25.5826588, 25.0543728

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2057151
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2057151
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8896875, 9.8881626, -4.1433077, 10.5423946, -14.4320822, 14.0314703
1: -10.4701576, 15.1473541, -11.1630354, 16.1227474, -26.5929012, 26.3103828
2: -6.7051382, 13.5473413, -7.1534371, 14.4338942, -21.1390324, 20.7007751
3: -11.7754116, 16.9464436, -12.5585098, 18.0531578, -29.8285675, 29.5049534
4: -8.0145988, 16.4772720, -8.5494747, 17.5393734, -25.5539703, 25.0267467

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2060852
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2050879, upper bound: 27.2060852
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9068065, 9.9074421, -4.3870020, 11.2383909, -15.1451969, 14.2944431
1: -10.5187998, 15.1827583, -11.8893881, 17.1601810, -27.6789780, 27.0721416
2: -6.7327509, 13.5722666, -7.6013036, 15.4227715, -22.1555176, 21.1735668
3: -11.8289604, 16.9825478, -13.3972139, 19.2366486, -31.0656090, 30.3797607
4: -8.0432854, 16.5048981, -9.0787373, 18.7438374, -26.7871227, 25.5836334

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2078943, upper bound: 27.2058551
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058905, upper bound: 27.2058551
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8896875, 9.8881626, -4.3870020, 11.2383909, -15.1280785, 14.2751627
1: -10.4701576, 15.1473541, -11.8893881, 17.1601810, -27.6303387, 27.0367393
2: -6.7051382, 13.5473413, -7.6013036, 15.4227715, -22.1279068, 21.1486416
3: -11.7754116, 16.9464436, -13.3972139, 19.2366486, -31.0120583, 30.3436584
4: -8.0145988, 16.4772720, -9.0787373, 18.7438374, -26.7584362, 25.5560093

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2078943, upper bound: 27.2062252
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2078943, upper bound: 27.2062252
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1111946, 10.4365177, -4.1433077, 10.5423946, -14.6535892, 14.5798254
1: -11.0838718, 15.9800167, -11.1630354, 16.1227474, -27.2066154, 27.1430473
2: -7.0965567, 14.2923832, -7.1534371, 14.4338942, -21.5304508, 21.4458199
3: -12.4648304, 17.8796692, -12.5585098, 18.0531578, -30.5179882, 30.4381790
4: -8.4751015, 17.3669186, -8.5494747, 17.5393734, -26.0144711, 25.9163914

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2064326, upper bound: 27.2065177
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2064326, upper bound: 27.2065177
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0895777, 10.4020500, -4.1433077, 10.5423946, -14.6319714, 14.5453577
1: -11.0181484, 15.9188347, -11.1630354, 16.1227474, -27.1408958, 27.0818634
2: -7.0597200, 14.2448359, -7.1534371, 14.4338942, -21.4936142, 21.3982677
3: -12.3929386, 17.8163891, -12.5585098, 18.0531578, -30.4460964, 30.3748989
4: -8.4366903, 17.3104019, -8.5494747, 17.5393734, -25.9760609, 25.8598766

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2064326, upper bound: 27.2065177
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2050879, upper bound: 27.2065177
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1111946, 10.4365177, -4.3870020, 11.2383909, -15.3495855, 14.8235197
1: -11.0838718, 15.9800167, -11.8893881, 17.1601810, -28.2440510, 27.8694038
2: -7.0965567, 14.2923832, -7.6013036, 15.4227715, -22.5193253, 21.8936863
3: -12.4648304, 17.8796692, -13.3972139, 19.2366486, -31.7014790, 31.2768822
4: -8.4751015, 17.3669186, -9.0787373, 18.7438374, -27.2189369, 26.4456520

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063607, upper bound: 27.2065476
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2078971, upper bound: 27.2064588
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1814336, upper bound: 27.1669081
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2080226, upper bound: 27.2065869
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0895777, 10.4020500, -4.3870020, 11.2383909, -15.3279676, 14.7890501
1: -11.0181484, 15.9188347, -11.8893881, 17.1601810, -28.1783295, 27.8082199
2: -7.0597200, 14.2448359, -7.6013036, 15.4227715, -22.4824867, 21.8461361
3: -12.3929386, 17.8163891, -13.3972139, 19.2366486, -31.6295872, 31.2136002
4: -8.4366903, 17.3104019, -9.0787373, 18.7438374, -27.1805267, 26.3891392

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063607, upper bound: 27.2065476
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063607, upper bound: 27.2065476
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1233888, 10.5656967, -3.9416430, 10.0241356, -14.1475239, 14.5073395
1: -11.1601391, 16.1530228, -10.6106949, 15.3450384, -26.5051765, 26.7637157
2: -7.1305256, 14.5066938, -6.7955890, 13.7306890, -20.8612137, 21.3022823
3: -12.5761127, 18.0918350, -11.9359779, 17.1758213, -29.7519341, 30.0278111
4: -8.5193834, 17.6468105, -8.1233921, 16.6995983, -25.2189827, 25.7701988

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1772885, upper bound: 27.1744776
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1942388, upper bound: 27.1990204
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1233888, 10.5656967, -4.1433077, 10.5423946, -14.6657829, 14.7090044
1: -11.1601391, 16.1530228, -11.1630354, 16.1227474, -27.2828827, 27.3160515
2: -7.1305256, 14.5066938, -7.1534371, 14.4338942, -21.5644188, 21.6601238
3: -12.5761127, 18.0918350, -12.5585098, 18.0531578, -30.6292706, 30.6503448
4: -8.5193834, 17.6468105, -8.5494747, 17.5393734, -26.0587578, 26.1962833

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1776629, upper bound: 27.1791080
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1946714, upper bound: 27.2000677
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.3296986, 11.0759592, -3.9416430, 10.0241356, -14.3538342, 15.0176020
1: -11.7401705, 16.9393539, -10.6106949, 15.3450384, -27.0852089, 27.5500469
2: -7.5014129, 15.2059689, -6.7955890, 13.7306890, -21.2321014, 22.0015583
3: -13.2249527, 18.9666977, -11.9359779, 17.1758213, -30.4007740, 30.9026718
4: -8.9525537, 18.4825020, -8.1233921, 16.6995983, -25.6521530, 26.6058941

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058551, upper bound: 27.2078943
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058551, upper bound: 27.2078943
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3330073, 11.0931511, -3.9416430, 10.0241356, -14.3571434, 15.0347939
1: -11.7429276, 16.9466839, -10.6106949, 15.3450384, -27.0879669, 27.5573788
2: -7.5071530, 15.2263012, -6.7955890, 13.7306890, -21.2378426, 22.0218906
3: -13.2298622, 18.9907055, -11.9359779, 17.1758213, -30.4056835, 30.9266777
4: -8.9654169, 18.5050793, -8.1233921, 16.6995983, -25.6650162, 26.6284714

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058551, upper bound: 27.2082814
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058551, upper bound: 27.2082814
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3296986, 11.0759592, -4.1433077, 10.5423946, -14.8720922, 15.2192669
1: -11.7401705, 16.9393539, -11.1630354, 16.1227474, -27.8629189, 28.1023808
2: -7.5014129, 15.2059689, -7.1534371, 14.4338942, -21.9353065, 22.3594036
3: -13.2249527, 18.9666977, -12.5585098, 18.0531578, -31.2781105, 31.5252075
4: -8.9525537, 18.4825020, -8.5494747, 17.5393734, -26.4919281, 27.0319767

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2066577, upper bound: 27.2085215
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058551, upper bound: 27.2085215
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3330073, 11.0931511, -4.1433077, 10.5423946, -14.8754025, 15.2364588
1: -11.7429276, 16.9466839, -11.1630354, 16.1227474, -27.8656712, 28.1097164
2: -7.5071530, 15.2263012, -7.1534371, 14.4338942, -21.9410477, 22.3797302
3: -13.2298622, 18.9907055, -12.5585098, 18.0531578, -31.2830200, 31.5492153
4: -8.9654169, 18.5050793, -8.5494747, 17.5393734, -26.5047913, 27.0545540

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058551, upper bound: 27.2089086
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2066577, upper bound: 27.2089086
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1233888, 10.5656967, -4.3870020, 11.2383909, -15.3617802, 14.9526968
1: -11.1601391, 16.1530228, -11.8893881, 17.1601810, -28.3203182, 28.0424099
2: -7.1305256, 14.5066938, -7.6013036, 15.4227715, -22.5532913, 22.1079941
3: -12.5761127, 18.0918350, -13.3972139, 19.2366486, -31.8127613, 31.4890480
4: -8.5193834, 17.6468105, -9.0787373, 18.7438374, -27.2632217, 26.7255459

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1750838, upper bound: 27.1652342
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1750838, upper bound: 27.2007646
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.3296986, 11.0759592, -4.1428089, 10.6177177, -14.9474163, 15.2187681
1: -11.7401705, 16.9393539, -11.2123156, 16.2294140, -27.9695854, 28.1516666
2: -7.5014129, 15.2059689, -7.1643281, 14.5773115, -22.0787239, 22.3702965
3: -13.2249527, 18.9666977, -12.6359673, 18.1798058, -31.4047585, 31.6026649
4: -8.9525537, 18.4825020, -8.5602360, 17.7325573, -26.6851120, 27.0427380

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2007158, upper bound: 27.1982637
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1821571, upper bound: 27.1686442
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1821571, upper bound: 27.1987273
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3330073, 11.0931511, -4.1428089, 10.6177177, -14.9507256, 15.2359600
1: -11.7429276, 16.9466839, -11.2123156, 16.2294140, -27.9723396, 28.1590004
2: -7.5071530, 15.2263012, -7.1643281, 14.5773115, -22.0844631, 22.3906288
3: -13.2298622, 18.9907055, -12.6359673, 18.1798058, -31.4096680, 31.6266727
4: -8.9654169, 18.5050793, -8.5602360, 17.7325573, -26.6979752, 27.0653152

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.1978040, upper bound: 27.1982585
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -27.1757178, upper bound: 27.1689419
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2002493, upper bound: 27.1991030
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.3296986, 11.0759592, -4.3870020, 11.2383909, -15.5680895, 15.4629583
1: -11.7401705, 16.9393539, -11.8893881, 17.1601810, -28.9003525, 28.8287373
2: -7.5014129, 15.2059689, -7.6013036, 15.4227715, -22.9241829, 22.8072720
3: -13.2249527, 18.9666977, -13.3972139, 19.2366486, -32.4616013, 32.3639107
4: -8.9525537, 18.4825020, -9.0787373, 18.7438374, -27.6963921, 27.5612392

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2080831, upper bound: 27.2086614
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2080831, upper bound: 27.2086614
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3330073, 11.0931511, -4.3870020, 11.2383909, -15.5713968, 15.4801502
1: -11.7429276, 16.9466839, -11.8893881, 17.1601810, -28.9031086, 28.8360710
2: -7.5071530, 15.2263012, -7.6013036, 15.4227715, -22.9299240, 22.8276005
3: -13.2298622, 18.9907055, -13.3972139, 19.2366486, -32.4665031, 32.3879204
4: -8.9654169, 18.5050793, -9.0787373, 18.7438374, -27.7092552, 27.5838146

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2067030, upper bound: 27.2084128
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2063778, upper bound: 27.2063777
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.2985473, 10.9991436, -4.5218935, 11.5179110, -15.8164577, 15.5210371
1: -11.6627293, 16.8279648, -12.1622829, 17.4638939, -29.1266232, 28.9902458
2: -7.4470334, 15.1158381, -7.7405324, 15.8200169, -23.2670479, 22.8563709
3: -13.1378975, 18.8385429, -13.7481842, 19.6917896, -32.8296890, 32.5867233
4: -8.8830223, 18.3745556, -9.2733250, 19.2468376, -28.1298599, 27.6478806

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2084825, upper bound: 27.2074174
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2094833, upper bound: 27.2078967
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.3938122, 11.2445555, -4.5218935, 11.5179110, -15.9117222, 15.7664490
1: -11.9078217, 17.1866608, -12.1622829, 17.4638939, -29.3717155, 29.3489418
2: -7.6145387, 15.4367199, -7.7405324, 15.8200169, -23.4345531, 23.1772518
3: -13.4159431, 19.2503071, -13.7481842, 19.6917896, -33.1077347, 32.9984894
4: -9.0873203, 18.7618866, -9.2733250, 19.2468376, -28.3341579, 28.0352097

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058958, upper bound: 27.1959394
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -27.2058958, upper bound: 27.2075371
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.2985473, 10.9991436, -4.5892553, 11.7021198, -16.0006676, 15.5883989
1: -11.6627293, 16.8279648, -12.3333263, 17.7274017, -29.3901310, 29.1612911
2: -7.4470334, 15.1158381, -7.8591914, 16.0541019, -23.5011330, 22.9750290
3: -13.1378975, 18.8385429, -13.9426527, 19.9966927, -33.1345863, 32.7811966
4: -8.8830223, 18.3745556, -9.4191265, 19.5313320, -28.4143543, 27.7936802

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.69 + 417.36 = 420.04 seconds
