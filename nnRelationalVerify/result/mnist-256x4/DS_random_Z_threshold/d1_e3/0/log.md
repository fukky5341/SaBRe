## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00162


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036630, 0.0036630)
1: (-0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009127, 0.0009127)
2: (-0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0048369, 0.0048369)
3: (0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0022016, 0.0022016)
4: (-0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009362, 0.0009362)
5: (-0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060836, 0.0060836)
6: (0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015441, 0.0015441)
7: (0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039950, 0.0039950)
8: (0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0021009, 0.0021009)
9: (-0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024361, 0.0024361)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.03 + 1.93 = 2.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0020250, upper bound: 0.0020250

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019309, upper bound: 0.0019391
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0019391, upper bound: 0.0019310
time: 1.14 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.50 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 0, lower bound: -0.0019309, upper bound: 0.0019391
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.50
Output dim: 0, lower bound: -0.0019391, upper bound: 0.0019310

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036315, 0.0036425
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009049, 0.0009076
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0048099, 0.0047953
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021826, 0.0021892
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009309, 0.0009281
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060495, 0.0060312
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015308, 0.0015354
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039606, 0.0039726
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020828, 0.0020892
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024225, 0.0024152

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018948, upper bound: 0.0018828
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018710, upper bound: 0.0019013
time: 1.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036630, 0.0036315
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009127, 0.0009049
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047953, 0.0048369
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0022016, 0.0021826
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009281, 0.0009362
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0060312, 0.0060836
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015441, 0.0015308
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039950, 0.0039606
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0021009, 0.0020828
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0024152, 0.0024361

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016395, upper bound: 0.0016395
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016395, upper bound: 0.0016395
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.80 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0018948, upper bound: 0.0018828
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0018710, upper bound: 0.0019013
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0016395, upper bound: 0.0016395
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.80
Output dim: 0, lower bound: -0.0016395, upper bound: 0.0016395

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035932, 0.0035780
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008953, 0.0008915
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047247, 0.0047448
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021596, 0.0021505
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009144, 0.0009184
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059424, 0.0059678
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015147, 0.0015082
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039189, 0.0039023
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020609, 0.0020522
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023796, 0.0023897

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018049, upper bound: 0.0018101
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018246, upper bound: 0.0017940
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035669, 0.0036015
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008888, 0.0008974
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047558, 0.0047101
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021438, 0.0021646
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009205, 0.0009116
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059815, 0.0059241
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015036, 0.0015182
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038902, 0.0039280
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020458, 0.0020657
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023953, 0.0023723

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018045, upper bound: 0.0018987
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018684, upper bound: 0.0018362
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036201, 0.0035882
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009020, 0.0008941
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047381, 0.0047803
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021758, 0.0021566
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009171, 0.0009252
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059593, 0.0060123
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015260, 0.0015125
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039482, 0.0039134
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020763, 0.0020580
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023864, 0.0024076

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016160, upper bound: 0.0016309
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016308, upper bound: 0.0016160
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0036630, 0.0035890
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0009127, 0.0008943
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0047393, 0.0048369
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0022016, 0.0021571
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009173, 0.0009362
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0059608, 0.0060836
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0015441, 0.0015129
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0039950, 0.0039144
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0021009, 0.0020585
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023870, 0.0024361

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015504, upper bound: 0.0015659
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015658, upper bound: 0.0015504
time: 0.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0018049, upper bound: 0.0018101
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0018246, upper bound: 0.0017940
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0018045, upper bound: 0.0018987
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0018684, upper bound: 0.0018362
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0016160, upper bound: 0.0016309
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0016308, upper bound: 0.0016160
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0015504, upper bound: 0.0015659
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0015658, upper bound: 0.0015504

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0033460, 0.0033550
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008337, 0.0008360
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0044303, 0.0044183
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0020110, 0.0020165
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008575, 0.0008552
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0055721, 0.0055571
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014105, 0.0014143
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0036493, 0.0036591
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0019191, 0.0019243
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0022313, 0.0022253

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017895, upper bound: 0.0018017
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017962, upper bound: 0.0017846
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0033727, 0.0033307
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008404, 0.0008299
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043982, 0.0044536
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0020271, 0.0020019
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008513, 0.0008620
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0055317, 0.0056014
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014217, 0.0014040
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0036784, 0.0036326
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0019344, 0.0019104
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0022152, 0.0022431

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018078, upper bound: 0.0017856
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0018161, upper bound: 0.0017713
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0034735, 0.0035480
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008655, 0.0008841
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046850, 0.0045867
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0020877, 0.0021324
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009068, 0.0008878
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0058925, 0.0057689
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014642, 0.0014956
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0037884, 0.0038695
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0019923, 0.0020350
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023596, 0.0023101

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017156, upper bound: 0.0018273
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017336, upper bound: 0.0018097
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035179, 0.0035081
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008766, 0.0008741
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046324, 0.0046454
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021144, 0.0021085
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008966, 0.0008991
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0058264, 0.0058427
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014829, 0.0014788
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038368, 0.0038261
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020177, 0.0020121
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023331, 0.0023397

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017534, upper bound: 0.0016207
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016447, upper bound: 0.0017235
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035356, 0.0035224
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008810, 0.0008777
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046513, 0.0046688
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021250, 0.0021171
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0009002, 0.0009036
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0058501, 0.0058721
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014904, 0.0014848
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038561, 0.0038417
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020279, 0.0020203
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023426, 0.0023514

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015235, upper bound: 0.0015565
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015412, upper bound: 0.0015410
time: 1.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0035559, 0.0035029
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008860, 0.0008728
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0046255, 0.0046955
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0021372, 0.0021053
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008953, 0.0009088
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0058176, 0.0059057
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014989, 0.0014766
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0038782, 0.0038204
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0020395, 0.0020091
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0023296, 0.0023649

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015410, upper bound: 0.0015412
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015565, upper bound: 0.0015235
time: 0.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0017895, upper bound: 0.0018017
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0017962, upper bound: 0.0017846
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0018078, upper bound: 0.0017856
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0018161, upper bound: 0.0017713
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0017156, upper bound: 0.0018273
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0017336, upper bound: 0.0018097
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0017534, upper bound: 0.0016207
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0016447, upper bound: 0.0017235
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0015235, upper bound: 0.0015565
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0015412, upper bound: 0.0015410
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0015410, upper bound: 0.0015412
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -0.0015565, upper bound: 0.0015235

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026603, 0.0026590
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006629, 0.0006625
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035111, 0.0035129
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015989, 0.0015981
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006796, 0.0006799
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0044161, 0.0044184
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011214, 0.0011209
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0029015, 0.0029000
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015259, 0.0015251
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017684, 0.0017693

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0015013
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0015013
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026336, 0.0026694
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006562, 0.0006651
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0035248, 0.0034776
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015829, 0.0016044
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006822, 0.0006731
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0044333, 0.0043739
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011101, 0.0011252
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028723, 0.0029113
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015105, 0.0015310
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017753, 0.0017515

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017386, upper bound: 0.0017821
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017937, upper bound: 0.0017120
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026870, 0.0026273
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006695, 0.0006547
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034693, 0.0035482
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016150, 0.0015791
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006715, 0.0006867
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0043635, 0.0044627
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011327, 0.0011075
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0029306, 0.0028655
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015412, 0.0015069
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017473, 0.0017871

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017849, upper bound: 0.0017767
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017990, upper bound: 0.0017593
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0026647, 0.0026451
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006640, 0.0006591
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034928, 0.0035187
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0016016, 0.0015898
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006760, 0.0006810
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0043930, 0.0044256
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0011233, 0.0011150
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0029062, 0.0028848
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0015283, 0.0015171
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017591, 0.0017722

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015199, upper bound: 0.0014744
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015199, upper bound: 0.0014744
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032243, 0.0033219
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008034, 0.0008277
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043866, 0.0042576
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019379, 0.0019966
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008490, 0.0008240
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0055171, 0.0053549
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013591, 0.0014003
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035165, 0.0036230
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018493, 0.0019053
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0022093, 0.0021443

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014453, upper bound: 0.0015238
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014453, upper bound: 0.0015238
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032495, 0.0032987
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008097, 0.0008219
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043559, 0.0042910
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019531, 0.0019826
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008431, 0.0008305
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0054786, 0.0053969
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013698, 0.0013905
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035441, 0.0035977
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018638, 0.0018920
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021939, 0.0021612

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017099, upper bound: 0.0018011
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017251, upper bound: 0.0017922
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0033352, 0.0032331
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008310, 0.0008056
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042693, 0.0044041
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0020046, 0.0019432
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008263, 0.0008524
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053696, 0.0055392
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0014059, 0.0013629
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0036375, 0.0035262
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0019129, 0.0018544
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021502, 0.0022181

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017322, upper bound: 0.0016120
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017443, upper bound: 0.0015921
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032429, 0.0033244
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008081, 0.0008283
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0043898, 0.0042823
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019491, 0.0019980
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008496, 0.0008288
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0055212, 0.0053860
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013670, 0.0014013
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035369, 0.0036257
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018600, 0.0019067
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0022109, 0.0021568

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016206, upper bound: 0.0017154
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016367, upper bound: 0.0017098
time: 1.10 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0015013
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0014944, upper bound: 0.0015013
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017386, upper bound: 0.0017821
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017937, upper bound: 0.0017120
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017849, upper bound: 0.0017767
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017990, upper bound: 0.0017593
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0015199, upper bound: 0.0014744
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0015199, upper bound: 0.0014744
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0014453, upper bound: 0.0015238
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0014453, upper bound: 0.0015238
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017099, upper bound: 0.0018011
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017251, upper bound: 0.0017922
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017322, upper bound: 0.0016120
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0017443, upper bound: 0.0015921
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0016206, upper bound: 0.0017154
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.20
Output dim: 0, lower bound: -0.0016367, upper bound: 0.0017098

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025038, 0.0025803
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006239, 0.0006429
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034073, 0.0033062
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015049, 0.0015508
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006595, 0.0006399
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042854, 0.0041584
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010554, 0.0010877
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027308, 0.0028142
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014361, 0.0014800
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017161, 0.0016652

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014674, upper bound: 0.0014871
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014674, upper bound: 0.0014871
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025454, 0.0025396
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006342, 0.0006328
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033535, 0.0033612
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015299, 0.0015264
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006491, 0.0006506
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042178, 0.0042275
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010730, 0.0010705
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027761, 0.0027698
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014599, 0.0014566
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016890, 0.0016929

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017659, upper bound: 0.0017028
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017848, upper bound: 0.0016976
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025057, 0.0024726
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006244, 0.0006161
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032650, 0.0033088
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015060, 0.0014861
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006319, 0.0006404
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0041065, 0.0041616
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010563, 0.0010423
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027329, 0.0026967
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014372, 0.0014182
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016444, 0.0016665

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016739, upper bound: 0.0015667
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015422, upper bound: 0.0016605
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025262, 0.0024460
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006295, 0.0006095
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032300, 0.0033358
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015183, 0.0014701
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006252, 0.0006456
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040624, 0.0041956
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010649, 0.0010311
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027552, 0.0026677
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014489, 0.0014029
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016268, 0.0016801

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016874, upper bound: 0.0015376
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015629, upper bound: 0.0016462
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025295, 0.0025585
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006303, 0.0006375
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033784, 0.0033402
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015203, 0.0015377
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006539, 0.0006465
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042492, 0.0042011
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010663, 0.0010785
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027588, 0.0027904
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014508, 0.0014674
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017016, 0.0016823

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 75

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015930, upper bound: 0.0015829
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015016, upper bound: 0.0016885
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025149, 0.0025787
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006267, 0.0006425
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034051, 0.0033209
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015116, 0.0015499
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006591, 0.0006428
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042828, 0.0041769
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010601, 0.0010870
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027429, 0.0028124
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014425, 0.0014790
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017150, 0.0016726

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014507, upper bound: 0.0014914
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014507, upper bound: 0.0014914
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0032993, 0.0032205
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008221, 0.0008025
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042527, 0.0043567
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019830, 0.0019356
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008231, 0.0008432
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053487, 0.0054796
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013908, 0.0013576
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0035984, 0.0035124
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0018923, 0.0018472
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021419, 0.0021943

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014029, upper bound: 0.0013390
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014029, upper bound: 0.0013390
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0033185, 0.0031972
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0008269, 0.0007967
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0042219, 0.0043820
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0019945, 0.0019216
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0008171, 0.0008481
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0053100, 0.0055114
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0013988, 0.0013477
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0036192, 0.0034870
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0019033, 0.0018338
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0021263, 0.0022070

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017203, upper bound: 0.0015838
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017363, upper bound: 0.0015746
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025677, 0.0026280
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006398, 0.0006548
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034702, 0.0033907
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015433, 0.0015795
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006716, 0.0006563
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0043646, 0.0042646
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010824, 0.0011078
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028005, 0.0028662
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014727, 0.0015073
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017478, 0.0017077

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0017063
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016119, upper bound: 0.0016952
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025525, 0.0026492
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006360, 0.0006601
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0034982, 0.0033705
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015341, 0.0015922
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006771, 0.0006524
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0043998, 0.0042392
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010760, 0.0011167
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027838, 0.0028893
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014640, 0.0015194
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017619, 0.0016976

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015474, upper bound: 0.0016380
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015647, upper bound: 0.0016219
time: 1.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0014674, upper bound: 0.0014871
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0014674, upper bound: 0.0014871
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0017659, upper bound: 0.0017028
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0017848, upper bound: 0.0016976
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0016739, upper bound: 0.0015667
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015422, upper bound: 0.0016605
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0016874, upper bound: 0.0015376
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015629, upper bound: 0.0016462
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015930, upper bound: 0.0015829
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015016, upper bound: 0.0016885
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0014507, upper bound: 0.0014914
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0014507, upper bound: 0.0014914
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0014029, upper bound: 0.0013390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0014029, upper bound: 0.0013390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0017203, upper bound: 0.0015838
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0017363, upper bound: 0.0015746
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015924, upper bound: 0.0017063
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0016119, upper bound: 0.0016952
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015474, upper bound: 0.0016380
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0015647, upper bound: 0.0016219

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024042, 0.0024147
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005991, 0.0006017
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031886, 0.0031747
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014450, 0.0014513
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006171, 0.0006144
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040104, 0.0039929
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010134, 0.0010179
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026221, 0.0026336
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013789, 0.0013850
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016059, 0.0015989

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014732, upper bound: 0.0014314
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014732, upper bound: 0.0014314
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024298, 0.0023983
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006055, 0.0005976
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031669, 0.0032086
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014604, 0.0014415
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006130, 0.0006210
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039832, 0.0040356
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010243, 0.0010110
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0026501, 0.0026157
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013937, 0.0013756
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015950, 0.0016160

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014280
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014280
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022781, 0.0021470
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005676, 0.0005350
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028351, 0.0030082
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013692, 0.0012904
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005487, 0.0005822
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0035658, 0.0037835
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009603, 0.0009050
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024846, 0.0023416
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013066, 0.0012314
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014279, 0.0015151

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013346, upper bound: 0.0012641
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013346, upper bound: 0.0012641
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021802, 0.0022275
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005432, 0.0005550
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029414, 0.0028789
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013104, 0.0013388
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005693, 0.0005572
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036995, 0.0036209
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009190, 0.0009390
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023778, 0.0024294
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012505, 0.0012776
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014814, 0.0014500

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0016580
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015394, upper bound: 0.0015820
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022949, 0.0021205
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005718, 0.0005284
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028001, 0.0030304
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013793, 0.0012745
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005420, 0.0005865
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0035218, 0.0038115
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009674, 0.0008939
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0025029, 0.0023127
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0013163, 0.0012162
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014103, 0.0015263

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016288, upper bound: 0.0015350
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016848, upper bound: 0.0014802
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022007, 0.0022091
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005483, 0.0005504
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029171, 0.0029060
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013227, 0.0013277
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005646, 0.0005624
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036689, 0.0036549
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009277, 0.0009312
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024001, 0.0024093
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012622, 0.0012670
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014692, 0.0014636

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015212, upper bound: 0.0016437
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015601, upper bound: 0.0015736
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022241, 0.0023502
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005542, 0.0005856
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031034, 0.0029369
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013367, 0.0014125
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006007, 0.0005684
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039033, 0.0036938
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009375, 0.0009907
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024257, 0.0025632
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012756, 0.0013480
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015630, 0.0014792

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014807, upper bound: 0.0016795
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014927, upper bound: 0.0016610
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025926, 0.0024457
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006460, 0.0006094
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032296, 0.0034235
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015582, 0.0014700
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006251, 0.0006626
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0040619, 0.0043059
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010929, 0.0010310
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028276, 0.0026674
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014870, 0.0014028
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016266, 0.0017243

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016331, upper bound: 0.0015154
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016473, upper bound: 0.0014947
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025777, 0.0024713
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006423, 0.0006158
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0032634, 0.0034039
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015493, 0.0014853
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006316, 0.0006588
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0041045, 0.0042812
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010866, 0.0010418
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0028114, 0.0026953
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014785, 0.0014175
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016436, 0.0017144

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014090, upper bound: 0.0013090
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0014090, upper bound: 0.0013090
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0024812, 0.0025623
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006182, 0.0006384
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033834, 0.0032764
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0014913, 0.0015400
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006549, 0.0006341
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042555, 0.0041208
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010459, 0.0010801
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027061, 0.0027945
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014231, 0.0014696
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0017041, 0.0016501

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013200, upper bound: 0.0013940
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013200, upper bound: 0.0013940
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0025052, 0.0025414
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0006242, 0.0006332
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0033559, 0.0033080
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0015057, 0.0015275
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006495, 0.0006403
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0042208, 0.0041606
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0010560, 0.0010713
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0027322, 0.0027717
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0014369, 0.0014576
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0016902, 0.0016661

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013842
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013842
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022280, 0.0023548
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005552, 0.0005868
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0031095, 0.0029420
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013391, 0.0014153
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0006018, 0.0005694
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0039110, 0.0037003
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009392, 0.0009926
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024299, 0.0025683
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012779, 0.0013506
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015661, 0.0014818

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015184, upper bound: 0.0016288
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015386, upper bound: 0.0016205
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022509, 0.0023246
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005609, 0.0005792
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0030697, 0.0029723
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013529, 0.0013972
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005941, 0.0005753
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0038608, 0.0037383
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009488, 0.0009799
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024549, 0.0025354
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012910, 0.0013333
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0015460, 0.0014970

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015395, upper bound: 0.0016127
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0015560, upper bound: 0.0016007
time: 1.08 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014732, upper bound: 0.0014314
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014732, upper bound: 0.0014314
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014280
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014926, upper bound: 0.0014280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0013346, upper bound: 0.0012641
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0013346, upper bound: 0.0012641
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015020, upper bound: 0.0016580
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015394, upper bound: 0.0015820
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0016288, upper bound: 0.0015350
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0016848, upper bound: 0.0014802
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015212, upper bound: 0.0016437
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015601, upper bound: 0.0015736
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014807, upper bound: 0.0016795
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014927, upper bound: 0.0016610
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0016331, upper bound: 0.0015154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0016473, upper bound: 0.0014947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014090, upper bound: 0.0013090
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0014090, upper bound: 0.0013090
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0013200, upper bound: 0.0013940
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0013200, upper bound: 0.0013940
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013842
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0013405, upper bound: 0.0013842
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015184, upper bound: 0.0016288
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015386, upper bound: 0.0016205
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015395, upper bound: 0.0016127
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.99
Output dim: 0, lower bound: -0.0015560, upper bound: 0.0016007

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020883, 0.0021864
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005204, 0.0005448
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028872, 0.0027576
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012551, 0.0013141
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005588, 0.0005337
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036313, 0.0034684
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008803, 0.0009217
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022776, 0.0023846
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011978, 0.0012540
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014541, 0.0013889

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022031, 0.0020793
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005489, 0.0005181
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027456, 0.0029091
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013241, 0.0012497
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005314, 0.0005631
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034533, 0.0036589
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009287, 0.0008765
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024027, 0.0022677
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012636, 0.0011926
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013829, 0.0014652

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0012400
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0012400
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022468, 0.0020286
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005598, 0.0005055
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0026788, 0.0029668
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013504, 0.0012193
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005185, 0.0005742
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0033692, 0.0037315
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009471, 0.0008551
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024504, 0.0022125
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012887, 0.0011635
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013492, 0.0014943

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013476, upper bound: 0.0012170
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013476, upper bound: 0.0012170
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021088, 0.0021705
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005255, 0.0005408
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028661, 0.0027847
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012675, 0.0013045
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005547, 0.0005390
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036048, 0.0035024
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008889, 0.0009149
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023000, 0.0023672
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012095, 0.0012449
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014435, 0.0014025

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012546, upper bound: 0.0013058
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012546, upper bound: 0.0013058
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020606, 0.0022057
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005135, 0.0005496
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029126, 0.0027210
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012385, 0.0013257
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005637, 0.0005267
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036632, 0.0034223
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008686, 0.0009298
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022474, 0.0024056
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011819, 0.0012651
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014669, 0.0013705

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012229, upper bound: 0.0013413
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012229, upper bound: 0.0013413
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020767, 0.0021868
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005175, 0.0005449
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028876, 0.0027423
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012482, 0.0013143
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005589, 0.0005308
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036318, 0.0034491
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008754, 0.0009218
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022650, 0.0023850
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011911, 0.0012542
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014543, 0.0013812

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0021912, 0.0020752
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005460, 0.0005171
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0027402, 0.0028935
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013170, 0.0012472
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005304, 0.0005600
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0034465, 0.0036392
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009237, 0.0008748
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0023898, 0.0022633
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012568, 0.0011902
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013801, 0.0014573

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013081, upper bound: 0.0012420
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013081, upper bound: 0.0012420
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0022133, 0.0020443
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005515, 0.0005094
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0026995, 0.0029227
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0013303, 0.0012287
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005225, 0.0005657
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0033953, 0.0036759
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0009330, 0.0008618
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0024139, 0.0022296
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0012695, 0.0011725
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0013596, 0.0014720

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020645, 0.0022083
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005144, 0.0005502
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0029160, 0.0027262
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012408, 0.0013273
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005644, 0.0005276
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036676, 0.0034288
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008703, 0.0009309
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022517, 0.0024085
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011841, 0.0012666
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014687, 0.0013730

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012400, upper bound: 0.0013125
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012400, upper bound: 0.0013125
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 1.0013143, 1.0059083, 1.0013143, 1.0059083, -0.0020877, 0.0021914
1: -0.0009365, 0.0002082, -0.0009365, 0.0002082, -0.0005202, 0.0005460
2: -0.0111574, -0.0050913, -0.0111574, -0.0050913, -0.0028937, 0.0027567
3: 0.0010442, 0.0038053, 0.0010442, 0.0038053, -0.0012548, 0.0013171
4: -0.0016316, -0.0004575, -0.0016316, -0.0004575, -0.0005601, 0.0005336
5: -0.0150737, -0.0074441, -0.0150737, -0.0074441, -0.0036395, 0.0034673
6: 0.0034302, 0.0053667, 0.0034302, 0.0053667, -0.0008800, 0.0009237
7: 0.0057374, 0.0107476, 0.0057374, 0.0107476, -0.0022769, 0.0023900
8: 0.0034531, 0.0060879, 0.0034531, 0.0060879, -0.0011974, 0.0012569
9: -0.0089231, -0.0058679, -0.0089231, -0.0058679, -0.0014574, 0.0013884

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
time: 1.00 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.02 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012356, upper bound: 0.0013215
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0012400
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013125, upper bound: 0.0012400
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013476, upper bound: 0.0012170
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013476, upper bound: 0.0012170
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012546, upper bound: 0.0013058
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012546, upper bound: 0.0013058
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012229, upper bound: 0.0013413
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012229, upper bound: 0.0013413
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012341, upper bound: 0.0013214
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013081, upper bound: 0.0012420
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013081, upper bound: 0.0012420
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012276
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012400, upper bound: 0.0013125
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012400, upper bound: 0.0013125
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.02
Output dim: 0, lower bound: -0.0012612, upper bound: 0.0013045

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.96 + 183.30 = 186.26 seconds
