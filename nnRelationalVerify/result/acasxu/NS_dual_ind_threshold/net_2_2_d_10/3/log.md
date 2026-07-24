## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.5360177700000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0888740, 0.5565974, -0.0888740, 0.5565974, -0.6454713, 0.6454713)
1: (-0.0582254, 0.5834452, -0.0582254, 0.5834452, -0.6416706, 0.6416706)
2: (-0.0401547, 0.5380218, -0.0401547, 0.5380218, -0.5781765, 0.5781765)
3: (-0.0950109, 0.5804102, -0.0950109, 0.5804102, -0.6754211, 0.6754211)
4: (-0.1025802, 0.5563258, -0.1025802, 0.5563258, -0.6589060, 0.6589060)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 0.93 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5955753, upper bound: 0.5955753

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5924615, upper bound: 0.5502826
time: 0.29 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.5924615, upper bound: 0.5502826
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0623974, 0.5213086, -0.0888740, 0.5565974, -0.6189947, 0.6101826
1: -0.0307785, 0.5498624, -0.0582254, 0.5834452, -0.6142237, 0.6080878
2: -0.0147544, 0.5086389, -0.0401547, 0.5380218, -0.5527762, 0.5487936
3: -0.0721118, 0.5497438, -0.0950109, 0.5804102, -0.6525221, 0.6447546
4: -0.0723428, 0.5184873, -0.1025802, 0.5563258, -0.6286685, 0.6210675

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
time: 0.29 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
time: 0.29 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0787928, 0.7256820, -0.0888740, 0.5565974, -0.6353902, 0.8145560
1: -0.0549843, 0.7266116, -0.0582254, 0.5834452, -0.6384295, 0.7848370
2: -0.0439436, 0.6844909, -0.0401547, 0.5380218, -0.5819654, 0.7246456
3: -0.0760016, 0.7251179, -0.0950109, 0.5804102, -0.6564119, 0.8201287
4: -0.0955775, 0.6710792, -0.1025802, 0.5563258, -0.6519033, 0.7736593

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.17 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -0.5496833, upper bound: 0.5496833

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0623974, 0.5213086, -0.0623974, 0.5213086, -0.5837060, 0.5837060
1: -0.0307785, 0.5498624, -0.0307785, 0.5498624, -0.5806409, 0.5806409
2: -0.0147544, 0.5086389, -0.0147544, 0.5086389, -0.5233933, 0.5233933
3: -0.0721118, 0.5497438, -0.0721118, 0.5497438, -0.6218556, 0.6218556
4: -0.0723428, 0.5184873, -0.0723428, 0.5184873, -0.5908301, 0.5908301

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5804298, upper bound: 0.5432507
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5921483, upper bound: 0.5456502
time: 0.32 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0623974, 0.5213086, -0.0787928, 0.7256820, -0.7880794, 0.6001014
1: -0.0307785, 0.5498624, -0.0549843, 0.7266116, -0.7573901, 0.6048467
2: -0.0147544, 0.5086389, -0.0439436, 0.6844909, -0.6992453, 0.5525825
3: -0.0721118, 0.5497438, -0.0760016, 0.7251179, -0.7972297, 0.6257454
4: -0.0723428, 0.5184873, -0.0955775, 0.6710792, -0.7434219, 0.6140648

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5804298, upper bound: 0.5432507
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5921483, upper bound: 0.5456502
time: 0.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0787928, 0.7256820, -0.0623974, 0.5213086, -0.6001014, 0.7880794
1: -0.0549843, 0.7266116, -0.0307785, 0.5498624, -0.6048467, 0.7573901
2: -0.0439436, 0.6844909, -0.0147544, 0.5086389, -0.5525825, 0.6992453
3: -0.0760016, 0.7251179, -0.0721118, 0.5497438, -0.6257454, 0.7972297
4: -0.0955775, 0.6710792, -0.0723428, 0.5184873, -0.6140648, 0.7434219

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496062, upper bound: 0.5432507
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5447122
time: 0.30 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0787928, 0.7256820, -0.0787928, 0.7256820, -0.8044748, 0.8044748
1: -0.0549843, 0.7266116, -0.0549843, 0.7266116, -0.7815959, 0.7815959
2: -0.0439436, 0.6844909, -0.0439436, 0.6844909, -0.7284346, 0.7284346
3: -0.0760016, 0.7251179, -0.0760016, 0.7251179, -0.8011195, 0.8011195
4: -0.0955775, 0.6710792, -0.0955775, 0.6710792, -0.7666566, 0.7666566

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5496062, upper bound: 0.5432507
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5447122
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.47 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5804298, upper bound: 0.5432507
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5921483, upper bound: 0.5456502
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5804298, upper bound: 0.5432507
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5921483, upper bound: 0.5456502
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5496062, upper bound: 0.5432507
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5447122
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5496062, upper bound: 0.5432507
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -0.5447122, upper bound: 0.5447122

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0559404, 0.5438366, -0.0623974, 0.5213086, -0.5772490, 0.6062340
1: -0.0274717, 0.5397282, -0.0307785, 0.5498624, -0.5773340, 0.5705067
2: -0.0103084, 0.5389469, -0.0147544, 0.5086389, -0.5189472, 0.5537013
3: -0.0746628, 0.5309802, -0.0721118, 0.5497438, -0.6244066, 0.6030921
4: -0.0700200, 0.5737555, -0.0723428, 0.5184873, -0.5885073, 0.6460983

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5789684
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5789684
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0544621, 0.5107545, -0.0623974, 0.5213086, -0.5757707, 0.5731518
1: -0.0231084, 0.5368993, -0.0307785, 0.5498624, -0.5729707, 0.5676779
2: -0.0079250, 0.5003921, -0.0147544, 0.5086389, -0.5165639, 0.5151465
3: -0.0656871, 0.5367917, -0.0721118, 0.5497438, -0.6154308, 0.6089035
4: -0.0655446, 0.5106149, -0.0723428, 0.5184873, -0.5840319, 0.5829576

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5813678
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5813678
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0559404, 0.5438366, -0.0787928, 0.7256820, -0.7816224, 0.6226294
1: -0.0274717, 0.5397282, -0.0549843, 0.7266116, -0.7540833, 0.5947125
2: -0.0103084, 0.5389469, -0.0439436, 0.6844909, -0.6947993, 0.5828905
3: -0.0746628, 0.5309802, -0.0760016, 0.7251179, -0.7997807, 0.6069819
4: -0.0700200, 0.5737555, -0.0955775, 0.6710792, -0.7410991, 0.6693330

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5432507
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5432507
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0544621, 0.5107545, -0.0787928, 0.7256820, -0.7801441, 0.5895473
1: -0.0231084, 0.5368993, -0.0549843, 0.7266116, -0.7497200, 0.5918837
2: -0.0079250, 0.5003921, -0.0439436, 0.6844909, -0.6924160, 0.5443357
3: -0.0656871, 0.5367917, -0.0760016, 0.7251179, -0.7908049, 0.6127933
4: -0.0655446, 0.5106149, -0.0955775, 0.6710792, -0.7366238, 0.6061924

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5456502
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5456502
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0521634, 0.6700391, -0.0623974, 0.5213086, -0.5734720, 0.7324364
1: -0.0260392, 0.6587738, -0.0307785, 0.5498624, -0.5759016, 0.6895523
2: -0.0169684, 0.6291288, -0.0147544, 0.5086389, -0.5256072, 0.6438832
3: -0.0610830, 0.6292216, -0.0721118, 0.5497438, -0.6108267, 0.7013335
4: -0.0660505, 0.6399950, -0.0723428, 0.5184873, -0.5845379, 0.7123378

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5789684
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5789684
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0729346, 0.7093088, -0.0623974, 0.5213086, -0.5942432, 0.7717062
1: -0.0502349, 0.7131226, -0.0307785, 0.5498624, -0.6000973, 0.7439011
2: -0.0388763, 0.6701704, -0.0147544, 0.5086389, -0.5475152, 0.6849248
3: -0.0710397, 0.7062358, -0.0721118, 0.5497438, -0.6207834, 0.7783476
4: -0.0899878, 0.6558931, -0.0723428, 0.5184873, -0.6084751, 0.7282359

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5804298
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5804298
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0521634, 0.6700391, -0.0787928, 0.7256820, -0.7778454, 0.7488319
1: -0.0260392, 0.6587738, -0.0549843, 0.7266116, -0.7526509, 0.7137581
2: -0.0169684, 0.6291288, -0.0439436, 0.6844909, -0.7014593, 0.6730725
3: -0.0610830, 0.6292216, -0.0760016, 0.7251179, -0.7862008, 0.7052233
4: -0.0660505, 0.6399950, -0.0955775, 0.6710792, -0.7371297, 0.7355725

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5432507
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5432507
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0729346, 0.7093088, -0.0787928, 0.7256820, -0.7986166, 0.7881016
1: -0.0502349, 0.7131226, -0.0549843, 0.7266116, -0.7768465, 0.7681069
2: -0.0388763, 0.6701704, -0.0439436, 0.6844909, -0.7233672, 0.7141141
3: -0.0710397, 0.7062358, -0.0760016, 0.7251179, -0.7961575, 0.7822374
4: -0.0899878, 0.6558931, -0.0955775, 0.6710792, -0.7610669, 0.7514706

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5447122
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5447122
time: 0.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.21 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5789684
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5789684
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5813678
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5813678
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5432507
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5789684, upper bound: 0.5432507
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5456502
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5906869, upper bound: 0.5456502
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5789684
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5789684
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5804298
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5804298
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5432507
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5432507
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5447122
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -0.5432507, upper bound: 0.5447122

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0559404, 0.5438366, -0.0559404, 0.5438366, -0.5997770, 0.5997770
1: -0.0274717, 0.5397282, -0.0274717, 0.5397282, -0.5671998, 0.5671998
2: -0.0103084, 0.5389469, -0.0103084, 0.5389469, -0.5492553, 0.5492553
3: -0.0746628, 0.5309802, -0.0746628, 0.5309802, -0.6056430, 0.6056430
4: -0.0700200, 0.5737555, -0.0700200, 0.5737555, -0.6437755, 0.6437755

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5546813
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5515078
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0559404, 0.5438366, -0.0544621, 0.5107545, -0.5666949, 0.5982987
1: -0.0274717, 0.5397282, -0.0231084, 0.5368993, -0.5643710, 0.5628365
2: -0.0103084, 0.5389469, -0.0079250, 0.5003921, -0.5107005, 0.5468719
3: -0.0746628, 0.5309802, -0.0656871, 0.5367917, -0.6114545, 0.5966673
4: -0.0700200, 0.5737555, -0.0655446, 0.5106149, -0.5806348, 0.6393001

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5546813
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5515078
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0544621, 0.5107545, -0.0559404, 0.5438366, -0.5982987, 0.5666949
1: -0.0231084, 0.5368993, -0.0274717, 0.5397282, -0.5628365, 0.5643710
2: -0.0079250, 0.5003921, -0.0103084, 0.5389469, -0.5468719, 0.5107005
3: -0.0656871, 0.5367917, -0.0746628, 0.5309802, -0.5966673, 0.6114545
4: -0.0655446, 0.5106149, -0.0700200, 0.5737555, -0.6393001, 0.5806348

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5903631, upper bound: 0.5766561
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5876333, upper bound: 0.5813603
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0544621, 0.5107545, -0.0544621, 0.5107545, -0.5652165, 0.5652165
1: -0.0231084, 0.5368993, -0.0231084, 0.5368993, -0.5600077, 0.5600077
2: -0.0079250, 0.5003921, -0.0079250, 0.5003921, -0.5083171, 0.5083171
3: -0.0656871, 0.5367917, -0.0656871, 0.5367917, -0.6024787, 0.6024787
4: -0.0655446, 0.5106149, -0.0655446, 0.5106149, -0.5761595, 0.5761595

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5903632, upper bound: 0.5867695
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5876333, upper bound: 0.5900252
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0559404, 0.5438366, -0.0521634, 0.6700391, -0.7259794, 0.5960000
1: -0.0274717, 0.5397282, -0.0260392, 0.6587738, -0.6862454, 0.5657674
2: -0.0103084, 0.5389469, -0.0169684, 0.6291288, -0.6394372, 0.5559152
3: -0.0746628, 0.5309802, -0.0610830, 0.6292216, -0.7038844, 0.5920632
4: -0.0700200, 0.5737555, -0.0660505, 0.6399950, -0.7100150, 0.6398060

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5346742
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5315008
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0559404, 0.5438366, -0.0729346, 0.7093088, -0.7652492, 0.6167712
1: -0.0274717, 0.5397282, -0.0502349, 0.7131226, -0.7405943, 0.5899631
2: -0.0103084, 0.5389469, -0.0388763, 0.6701704, -0.6804788, 0.5778232
3: -0.0746628, 0.5309802, -0.0710397, 0.7062358, -0.7808986, 0.6020199
4: -0.0700200, 0.5737555, -0.0899878, 0.6558931, -0.7259131, 0.6637433

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5346742
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5315008
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0544621, 0.5107545, -0.0521634, 0.6700391, -0.7245011, 0.5629178
1: -0.0231084, 0.5368993, -0.0260392, 0.6587738, -0.6818821, 0.5629386
2: -0.0079250, 0.5003921, -0.0169684, 0.6291288, -0.6370538, 0.5173604
3: -0.0656871, 0.5367917, -0.0610830, 0.6292216, -0.6949087, 0.5978746
4: -0.0655446, 0.5106149, -0.0660505, 0.6399950, -0.7055396, 0.5766654

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5872666, upper bound: 0.5370322
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5729201, upper bound: 0.5344230
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0544621, 0.5107545, -0.0729346, 0.7093088, -0.7637709, 0.5836891
1: -0.0231084, 0.5368993, -0.0502349, 0.7131226, -0.7362310, 0.5871342
2: -0.0079250, 0.5003921, -0.0388763, 0.6701704, -0.6780955, 0.5392684
3: -0.0656871, 0.5367917, -0.0710397, 0.7062358, -0.7719228, 0.6078314
4: -0.0655446, 0.5106149, -0.0899878, 0.6558931, -0.7214378, 0.6006026

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5872666, upper bound: 0.5370322
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5729201, upper bound: 0.5344230
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0521634, 0.6700391, -0.0559404, 0.5438366, -0.5960000, 0.7259794
1: -0.0260392, 0.6587738, -0.0274717, 0.5397282, -0.5657674, 0.6862454
2: -0.0169684, 0.6291288, -0.0103084, 0.5389469, -0.5559152, 0.6394372
3: -0.0610830, 0.6292216, -0.0746628, 0.5309802, -0.5920632, 0.7038844
4: -0.0660505, 0.6399950, -0.0700200, 0.5737555, -0.6398060, 0.7100150

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5546813
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5515078
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0521634, 0.6700391, -0.0544621, 0.5107545, -0.5629178, 0.7245011
1: -0.0260392, 0.6587738, -0.0231084, 0.5368993, -0.5629386, 0.6818821
2: -0.0169684, 0.6291288, -0.0079250, 0.5003921, -0.5173604, 0.6370538
3: -0.0610830, 0.6292216, -0.0656871, 0.5367917, -0.5978746, 0.6949087
4: -0.0660505, 0.6399950, -0.0655446, 0.5106149, -0.5766654, 0.7055396

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5546813
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5515078
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0729346, 0.7093088, -0.0559404, 0.5438366, -0.6167712, 0.7652492
1: -0.0502349, 0.7131226, -0.0274717, 0.5397282, -0.5899631, 0.7405943
2: -0.0388763, 0.6701704, -0.0103084, 0.5389469, -0.5778232, 0.6804788
3: -0.0710397, 0.7062358, -0.0746628, 0.5309802, -0.6020199, 0.7808986
4: -0.0899878, 0.6558931, -0.0700200, 0.5737555, -0.6637433, 0.7259131

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5560856
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5524521
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0729346, 0.7093088, -0.0544621, 0.5107545, -0.5836891, 0.7637709
1: -0.0502349, 0.7131226, -0.0231084, 0.5368993, -0.5871342, 0.7362310
2: -0.0388763, 0.6701704, -0.0079250, 0.5003921, -0.5392684, 0.6780955
3: -0.0710397, 0.7062358, -0.0656871, 0.5367917, -0.6078314, 0.7719228
4: -0.0899878, 0.6558931, -0.0655446, 0.5106149, -0.6006026, 0.7214378

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5624218
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5559193
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0521634, 0.6700391, -0.0521634, 0.6700391, -0.7222024, 0.7222024
1: -0.0260392, 0.6587738, -0.0260392, 0.6587738, -0.6848130, 0.6848130
2: -0.0169684, 0.6291288, -0.0169684, 0.6291288, -0.6460972, 0.6460972
3: -0.0610830, 0.6292216, -0.0610830, 0.6292216, -0.6903046, 0.6903046
4: -0.0660505, 0.6399950, -0.0660505, 0.6399950, -0.7060456, 0.7060456

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5346742
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5315008
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0521634, 0.6700391, -0.0729346, 0.7093088, -0.7614722, 0.7429737
1: -0.0260392, 0.6587738, -0.0502349, 0.7131226, -0.7391618, 0.7090087
2: -0.0169684, 0.6291288, -0.0388763, 0.6701704, -0.6871388, 0.6680051
3: -0.0610830, 0.6292216, -0.0710397, 0.7062358, -0.7673187, 0.7002613
4: -0.0660505, 0.6399950, -0.0899878, 0.6558931, -0.7219437, 0.7299828

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5346742
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5315008
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0729346, 0.7093088, -0.0521634, 0.6700391, -0.7429737, 0.7614722
1: -0.0502349, 0.7131226, -0.0260392, 0.6587738, -0.7090087, 0.7391618
2: -0.0388763, 0.6701704, -0.0169684, 0.6291288, -0.6680051, 0.6871388
3: -0.0710397, 0.7062358, -0.0610830, 0.6292216, -0.7002613, 0.7673187
4: -0.0899878, 0.6558931, -0.0660505, 0.6399950, -0.7299828, 0.7219437

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5360786
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5324451
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0729346, 0.7093088, -0.0729346, 0.7093088, -0.7822434, 0.7822434
1: -0.0502349, 0.7131226, -0.0502349, 0.7131226, -0.7633575, 0.7633575
2: -0.0388763, 0.6701704, -0.0388763, 0.6701704, -0.7090467, 0.7090467
3: -0.0710397, 0.7062358, -0.0710397, 0.7062358, -0.7772754, 0.7772754
4: -0.0899878, 0.6558931, -0.0899878, 0.6558931, -0.7458809, 0.7458809

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5360786
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5324451
time: 0.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.22 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5546813
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5515078
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5546813
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5515078
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5903631, upper bound: 0.5766561
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5876333, upper bound: 0.5813603
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5903632, upper bound: 0.5867695
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5876333, upper bound: 0.5900252
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5346742
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5315008
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5754113, upper bound: 0.5346742
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5515079, upper bound: 0.5315008
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5872666, upper bound: 0.5370322
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5729201, upper bound: 0.5344230
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5872666, upper bound: 0.5370322
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5729201, upper bound: 0.5344230
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5546813
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5515078
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5546813
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5515078
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5560856
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5524521
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5624218
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5559193
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5346742
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5315008
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5445984, upper bound: 0.5346742
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5335492, upper bound: 0.5315008
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5360786
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5324451
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5396735, upper bound: 0.5360786
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -0.5315008, upper bound: 0.5324451

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0559404, 0.5438366, -0.5948964, 0.5907413
1: -0.0214454, 0.5362778, -0.0274717, 0.5397282, -0.5611736, 0.5637494
2: -0.0051125, 0.5333415, -0.0103084, 0.5389469, -0.5440594, 0.5436499
3: -0.0692544, 0.5269960, -0.0746628, 0.5309802, -0.6002346, 0.6016588
4: -0.0645348, 0.5658602, -0.0700200, 0.5737555, -0.6382903, 0.6358802

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5514244
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5514244
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0559404, 0.5438366, -0.6264254, 0.6218513
1: -0.0585842, 0.5600882, -0.0274717, 0.5397282, -0.5983124, 0.5875598
2: -0.0418383, 0.5510719, -0.0103084, 0.5389469, -0.5807852, 0.5613803
3: -0.0992109, 0.5518441, -0.0746628, 0.5309802, -0.6301911, 0.6265069
4: -0.1008286, 0.5853299, -0.0700200, 0.5737555, -0.6745842, 0.6553499

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5515079
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5515079
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0544621, 0.5107545, -0.5618143, 0.5892630
1: -0.0214454, 0.5362778, -0.0231084, 0.5368993, -0.5583447, 0.5593861
2: -0.0051125, 0.5333415, -0.0079250, 0.5003921, -0.5055046, 0.5412666
3: -0.0692544, 0.5269960, -0.0656871, 0.5367917, -0.6060461, 0.5926830
4: -0.0645348, 0.5658602, -0.0655446, 0.5106149, -0.5751497, 0.6314048

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5728257
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5728257
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0544621, 0.5107545, -0.5933433, 0.6203730
1: -0.0585842, 0.5600882, -0.0231084, 0.5368993, -0.5954835, 0.5831965
2: -0.0418383, 0.5510719, -0.0079250, 0.5003921, -0.5422304, 0.5589970
3: -0.0992109, 0.5518441, -0.0656871, 0.5367917, -0.6360025, 0.6175311
4: -0.1008286, 0.5853299, -0.0655446, 0.5106149, -0.6114435, 0.6508745

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5729201
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5729201
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0168777, 0.4548433, -0.0559404, 0.5438366, -0.5269589, 0.5107837
1: 0.0373445, 0.5012081, -0.0274717, 0.5397282, -0.5023837, 0.5286797
2: 0.0585996, 0.4470065, -0.0103084, 0.5389469, -0.4803473, 0.4573149
3: -0.0116420, 0.4992789, -0.0746628, 0.5309802, -0.5426222, 0.5739417
4: 0.0119382, 0.4438587, -0.0700200, 0.5737555, -0.5618173, 0.5138787

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5750752, upper bound: 0.5477958
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720192, upper bound: 0.5399059
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0300477, 0.4813079, -0.0559404, 0.5438366, -0.5738842, 0.5372483
1: -0.0013833, 0.5196805, -0.0274717, 0.5397282, -0.5411115, 0.5471522
2: 0.0164753, 0.4788888, -0.0103084, 0.5389469, -0.5224715, 0.4891972
3: -0.0416090, 0.5189773, -0.0746628, 0.5309802, -0.5725893, 0.5936401
4: -0.0368971, 0.4754637, -0.0700200, 0.5737555, -0.6106526, 0.5454836

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5742268, upper bound: 0.5783335
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710533, upper bound: 0.5544301
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0168777, 0.4548433, -0.0544621, 0.5107545, -0.4938768, 0.5093054
1: 0.0373445, 0.5012081, -0.0231084, 0.5368993, -0.4995549, 0.5243164
2: 0.0585996, 0.4470065, -0.0079250, 0.5003921, -0.4417925, 0.4549316
3: -0.0116420, 0.4992789, -0.0656871, 0.5367917, -0.5484337, 0.5649660
4: 0.0119382, 0.4438587, -0.0655446, 0.5106149, -0.4986767, 0.5094033

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5867694
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5867694
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0300477, 0.4813079, -0.0544621, 0.5107545, -0.5408021, 0.5357699
1: -0.0013833, 0.5196805, -0.0231084, 0.5368993, -0.5382826, 0.5427889
2: 0.0164753, 0.4788888, -0.0079250, 0.5003921, -0.4839168, 0.4868139
3: -0.0416090, 0.5189773, -0.0656871, 0.5367917, -0.5784007, 0.5846643
4: -0.0368971, 0.4754637, -0.0655446, 0.5106149, -0.5475119, 0.5410083

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5900252
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5900252
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0521634, 0.6700391, -0.7210989, 0.5869642
1: -0.0214454, 0.5362778, -0.0260392, 0.6587738, -0.6802192, 0.5623170
2: -0.0051125, 0.5333415, -0.0169684, 0.6291288, -0.6342413, 0.5503099
3: -0.0692544, 0.5269960, -0.0610830, 0.6292216, -0.6984760, 0.5880789
4: -0.0645348, 0.5658602, -0.0660505, 0.6399950, -0.7045298, 0.6319107

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5334657
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5334657
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0521634, 0.6700391, -0.7526278, 0.6180743
1: -0.0585842, 0.5600882, -0.0260392, 0.6587738, -0.7173580, 0.5861274
2: -0.0418383, 0.5510719, -0.0169684, 0.6291288, -0.6709671, 0.5680403
3: -0.0992109, 0.5518441, -0.0610830, 0.6292216, -0.7284325, 0.6129270
4: -0.1008286, 0.5853299, -0.0660505, 0.6399950, -0.7408237, 0.6513804

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5335492
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5335492
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0729346, 0.7093088, -0.7603686, 0.6077355
1: -0.0214454, 0.5362778, -0.0502349, 0.7131226, -0.7345680, 0.5865127
2: -0.0051125, 0.5333415, -0.0388763, 0.6701704, -0.6752830, 0.5722178
3: -0.0692544, 0.5269960, -0.0710397, 0.7062358, -0.7754902, 0.5980356
4: -0.0645348, 0.5658602, -0.0899878, 0.6558931, -0.7204279, 0.6558480

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5314173
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524521, upper bound: 0.5314173
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0729346, 0.7093088, -0.7918976, 0.6388456
1: -0.0585842, 0.5600882, -0.0502349, 0.7131226, -0.7717068, 0.6103231
2: -0.0418383, 0.5510719, -0.0388763, 0.6701704, -0.7120087, 0.5899482
3: -0.0992109, 0.5518441, -0.0710397, 0.7062358, -0.8054466, 0.6228837
4: -0.1008286, 0.5853299, -0.0899878, 0.6558931, -0.7567218, 0.6753176

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524521, upper bound: 0.5315008
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524521, upper bound: 0.5315008
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0483578, 0.5017393, -0.0521634, 0.6700391, -0.7183969, 0.5539027
1: -0.0165561, 0.5322876, -0.0260392, 0.6587738, -0.6753299, 0.5583268
2: -0.0020304, 0.4945726, -0.0169684, 0.6291288, -0.6311592, 0.5115409
3: -0.0596782, 0.5316396, -0.0610830, 0.6292216, -0.6888998, 0.5927225
4: -0.0589769, 0.5026535, -0.0660505, 0.6399950, -0.6989719, 0.5687040

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358641
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358641
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0551119, 0.5142183, -0.0521634, 0.6700391, -0.7251509, 0.5663816
1: -0.0310310, 0.5400121, -0.0260392, 0.6587738, -0.6898048, 0.5660514
2: -0.0154766, 0.5017341, -0.0169684, 0.6291288, -0.6446054, 0.5187024
3: -0.0722581, 0.5358996, -0.0610830, 0.6292216, -0.7014797, 0.5969826
4: -0.0708025, 0.5060241, -0.0660505, 0.6399950, -0.7107975, 0.5720746

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358683
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358683
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0483578, 0.5017393, -0.0729346, 0.7093088, -0.7576666, 0.5746740
1: -0.0165561, 0.5322876, -0.0502349, 0.7131226, -0.7296788, 0.5825225
2: -0.0020304, 0.4945726, -0.0388763, 0.6701704, -0.6722008, 0.5334488
3: -0.0596782, 0.5316396, -0.0710397, 0.7062358, -0.7659140, 0.6026793
4: -0.0589769, 0.5026535, -0.0899878, 0.6558931, -0.7148700, 0.5926412

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5343178
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5343178
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0551119, 0.5142183, -0.0729346, 0.7093088, -0.7644207, 0.5871529
1: -0.0310310, 0.5400121, -0.0502349, 0.7131226, -0.7441536, 0.5902470
2: -0.0154766, 0.5017341, -0.0388763, 0.6701704, -0.6856470, 0.5406104
3: -0.0722581, 0.5358996, -0.0710397, 0.7062358, -0.7784939, 0.6069393
4: -0.0708025, 0.5060241, -0.0899878, 0.6558931, -0.7266956, 0.5960118

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5344231
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5344231
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0559404, 0.5438366, -0.5903761, 0.7100296
1: -0.0209305, 0.6510010, -0.0274717, 0.5397282, -0.5606587, 0.6784726
2: -0.0117567, 0.6184078, -0.0103084, 0.5389469, -0.5507035, 0.6287162
3: -0.0549345, 0.6207125, -0.0746628, 0.5309802, -0.5859147, 0.6953753
4: -0.0604364, 0.6286552, -0.0700200, 0.5737555, -0.6341919, 0.6986752

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0363288, 0.6410381, -0.0559404, 0.5438366, -0.5801654, 0.6969784
1: -0.0170506, 0.6463324, -0.0274717, 0.5397282, -0.5567788, 0.6738040
2: -0.0088149, 0.6185240, -0.0103084, 0.5389469, -0.5477618, 0.6288324
3: -0.0517277, 0.6142127, -0.0746628, 0.5309802, -0.5827079, 0.6888755
4: -0.0549340, 0.6248241, -0.0700200, 0.5737555, -0.6286895, 0.6948441

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5515079
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5515079
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0544621, 0.5107545, -0.5572940, 0.7085513
1: -0.0209305, 0.6510010, -0.0231084, 0.5368993, -0.5578299, 0.6741093
2: -0.0117567, 0.6184078, -0.0079250, 0.5003921, -0.5121487, 0.6263329
3: -0.0549345, 0.6207125, -0.0656871, 0.5367917, -0.5917262, 0.6863995
4: -0.0604364, 0.6286552, -0.0655446, 0.5106149, -0.5710512, 0.6941998

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0363288, 0.6410381, -0.0544621, 0.5107545, -0.5470833, 0.6955001
1: -0.0170506, 0.6463324, -0.0231084, 0.5368993, -0.5539500, 0.6694407
2: -0.0088149, 0.6185240, -0.0079250, 0.5003921, -0.5092070, 0.6264490
3: -0.0517277, 0.6142127, -0.0656871, 0.5367917, -0.5885193, 0.6798998
4: -0.0549340, 0.6248241, -0.0655446, 0.5106149, -0.5655489, 0.6903687

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5358641, upper bound: 0.5729201
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5358641, upper bound: 0.5729201
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0559404, 0.5438366, -0.6114899, 0.7470059
1: -0.0455999, 0.7031308, -0.0274717, 0.5397282, -0.5853281, 0.7306025
2: -0.0339785, 0.6572912, -0.0103084, 0.5389469, -0.5729253, 0.6675996
3: -0.0652640, 0.6937898, -0.0746628, 0.5309802, -0.5962442, 0.7684526
4: -0.0847011, 0.6410571, -0.0700200, 0.5737555, -0.6584566, 0.7110770

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0566970, 0.6792859, -0.0559404, 0.5438366, -0.6005336, 0.7352263
1: -0.0404696, 0.7016609, -0.0274717, 0.5397282, -0.5801978, 0.7291325
2: -0.0304444, 0.6600475, -0.0103084, 0.5389469, -0.5693913, 0.6703559
3: -0.0613841, 0.6990961, -0.0746628, 0.5309802, -0.5923643, 0.7737589
4: -0.0785533, 0.6437433, -0.0700200, 0.5737555, -0.6523088, 0.7137633

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0544621, 0.5107545, -0.5784078, 0.7455276
1: -0.0455999, 0.7031308, -0.0231084, 0.5368993, -0.5824993, 0.7262392
2: -0.0339785, 0.6572912, -0.0079250, 0.5003921, -0.5343705, 0.6652163
3: -0.0652640, 0.6937898, -0.0656871, 0.5367917, -0.6020557, 0.7594769
4: -0.0847011, 0.6410571, -0.0655446, 0.5106149, -0.5953159, 0.7066017

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0566970, 0.6792859, -0.0544621, 0.5107545, -0.5674515, 0.7337480
1: -0.0404696, 0.7016609, -0.0231084, 0.5368993, -0.5773690, 0.7247692
2: -0.0304444, 0.6600475, -0.0079250, 0.5003921, -0.5308365, 0.6679726
3: -0.0613841, 0.6990961, -0.0656871, 0.5367917, -0.5981758, 0.7647831
4: -0.0785533, 0.6437433, -0.0655446, 0.5106149, -0.5891682, 0.7092879

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0521634, 0.6700391, -0.7165786, 0.7062526
1: -0.0209305, 0.6510010, -0.0260392, 0.6587738, -0.6797043, 0.6770402
2: -0.0117567, 0.6184078, -0.0169684, 0.6291288, -0.6408855, 0.6353762
3: -0.0549345, 0.6207125, -0.0610830, 0.6292216, -0.6841561, 0.6817954
4: -0.0604364, 0.6286552, -0.0660505, 0.6399950, -0.7004314, 0.6947057

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5334657
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5334657
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0729346, 0.7093088, -0.7558483, 0.7270238
1: -0.0209305, 0.6510010, -0.0502349, 0.7131226, -0.7340531, 0.7012359
2: -0.0117567, 0.6184078, -0.0388763, 0.6701704, -0.6819271, 0.6572841
3: -0.0549345, 0.6207125, -0.0710397, 0.7062358, -0.7611703, 0.6917521
4: -0.0604364, 0.6286552, -0.0899878, 0.6558931, -0.7163295, 0.7186430

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5344935, upper bound: 0.5314173
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5344935, upper bound: 0.5314173
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0521634, 0.6700391, -0.7376924, 0.7432289
1: -0.0455999, 0.7031308, -0.0260392, 0.6587738, -0.7043737, 0.7291701
2: -0.0339785, 0.6572912, -0.0169684, 0.6291288, -0.6631073, 0.6742596
3: -0.0652640, 0.6937898, -0.0610830, 0.6292216, -0.6944856, 0.7548728
4: -0.0847011, 0.6410571, -0.0660505, 0.6399950, -0.7246961, 0.7071076

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5344935
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5344935
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0729346, 0.7093088, -0.7769621, 0.7640001
1: -0.0455999, 0.7031308, -0.0502349, 0.7131226, -0.7587225, 0.7533658
2: -0.0339785, 0.6572912, -0.0388763, 0.6701704, -0.7041489, 0.6961675
3: -0.0652640, 0.6937898, -0.0710397, 0.7062358, -0.7714998, 0.7648295
4: -0.0847011, 0.6410571, -0.0899878, 0.6558931, -0.7405942, 0.7310448

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5324451
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5324451
time: 0.35 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.40 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5514244
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5514244
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5515079
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5515079
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5728257
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5728257
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5729201
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5543248, upper bound: 0.5729201
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5750752, upper bound: 0.5477958
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5720192, upper bound: 0.5399059
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5742268, upper bound: 0.5783335
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5710533, upper bound: 0.5544301
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5867694
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5867694
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5900252
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5805064, upper bound: 0.5900252
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5334657
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5334657
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5335492
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5335492
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5514244, upper bound: 0.5314173
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5524521, upper bound: 0.5314173
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5524521, upper bound: 0.5315008
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5524521, upper bound: 0.5315008
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358641
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358641
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358683
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5728257, upper bound: 0.5358683
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5343178
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5343178
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5344231
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5748092, upper bound: 0.5344231
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5515079
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5515079
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5514244
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5358641, upper bound: 0.5729201
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5358641, upper bound: 0.5729201
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5524521
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5559193
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5334657
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5334657, upper bound: 0.5334657
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5344935, upper bound: 0.5314173
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5344935, upper bound: 0.5314173
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5344935
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5314173, upper bound: 0.5344935
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5324451
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -0.5318548, upper bound: 0.5324451

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0510598, 0.5348009, -0.5858607, 0.5858607
1: -0.0214454, 0.5362778, -0.0214454, 0.5362778, -0.5577232, 0.5577232
2: -0.0051125, 0.5333415, -0.0051125, 0.5333415, -0.5384541, 0.5384541
3: -0.0692544, 0.5269960, -0.0692544, 0.5269960, -0.5962504, 0.5962504
4: -0.0645348, 0.5658602, -0.0645348, 0.5658602, -0.6303950, 0.6303950

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0825888, 0.5659109, -0.6169708, 0.6173897
1: -0.0214454, 0.5362778, -0.0585842, 0.5600882, -0.5815336, 0.5948620
2: -0.0051125, 0.5333415, -0.0418383, 0.5510719, -0.5561845, 0.5751798
3: -0.0692544, 0.5269960, -0.0992109, 0.5518441, -0.6210985, 0.6262068
4: -0.0645348, 0.5658602, -0.1008286, 0.5853299, -0.6498647, 0.6666889

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0510598, 0.5348009, -0.6173897, 0.6169708
1: -0.0585842, 0.5600882, -0.0214454, 0.5362778, -0.5948620, 0.5815336
2: -0.0418383, 0.5510719, -0.0051125, 0.5333415, -0.5751798, 0.5561845
3: -0.0992109, 0.5518441, -0.0692544, 0.5269960, -0.6262068, 0.6210985
4: -0.1008286, 0.5853299, -0.0645348, 0.5658602, -0.6666889, 0.6498647

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0825888, 0.5659109, -0.6484997, 0.6484997
1: -0.0585842, 0.5600882, -0.0585842, 0.5600882, -0.6186724, 0.6186724
2: -0.0418383, 0.5510719, -0.0418383, 0.5510719, -0.5929102, 0.5929102
3: -0.0992109, 0.5518441, -0.0992109, 0.5518441, -0.6510549, 0.6510549
4: -0.1008286, 0.5853299, -0.1008286, 0.5853299, -0.6861585, 0.6861585

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0483578, 0.5017393, -0.5527992, 0.5831587
1: -0.0214454, 0.5362778, -0.0165561, 0.5322876, -0.5537330, 0.5528339
2: -0.0051125, 0.5333415, -0.0020304, 0.4945726, -0.4996851, 0.5353719
3: -0.0692544, 0.5269960, -0.0596782, 0.5316396, -0.6008940, 0.5866742
4: -0.0645348, 0.5658602, -0.0589769, 0.5026535, -0.5671883, 0.6248371

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0551119, 0.5142183, -0.5652781, 0.5899128
1: -0.0214454, 0.5362778, -0.0310310, 0.5400121, -0.5614575, 0.5673088
2: -0.0051125, 0.5333415, -0.0154766, 0.5017341, -0.5068466, 0.5488181
3: -0.0692544, 0.5269960, -0.0722581, 0.5358996, -0.6051540, 0.5992541
4: -0.0645348, 0.5658602, -0.0708025, 0.5060241, -0.5705588, 0.6366627

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0483578, 0.5017393, -0.5843281, 0.6142688
1: -0.0585842, 0.5600882, -0.0165561, 0.5322876, -0.5908718, 0.5766443
2: -0.0418383, 0.5510719, -0.0020304, 0.4945726, -0.5364108, 0.5531023
3: -0.0992109, 0.5518441, -0.0596782, 0.5316396, -0.6308504, 0.6115223
4: -0.1008286, 0.5853299, -0.0589769, 0.5026535, -0.6034821, 0.6443068

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0551119, 0.5142183, -0.5968071, 0.6210228
1: -0.0585842, 0.5600882, -0.0310310, 0.5400121, -0.5985963, 0.5911192
2: -0.0418383, 0.5510719, -0.0154766, 0.5017341, -0.5435724, 0.5665485
3: -0.0992109, 0.5518441, -0.0722581, 0.5358996, -0.6351105, 0.6241022
4: -0.1008286, 0.5853299, -0.0708025, 0.5060241, -0.6068527, 0.6561323

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0168777, 0.4548433, -0.0510598, 0.5348009, -0.5179232, 0.5059032
1: 0.0373445, 0.5012081, -0.0214454, 0.5362778, -0.4989333, 0.5226535
2: 0.0585996, 0.4470065, -0.0051125, 0.5333415, -0.4747419, 0.4521191
3: -0.0116420, 0.4992789, -0.0692544, 0.5269960, -0.5386379, 0.5685333
4: 0.0119382, 0.4438587, -0.0645348, 0.5658602, -0.5539220, 0.5083935

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719236, upper bound: 0.5399059
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5719236, upper bound: 0.5399059
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0168777, 0.4548433, -0.0825888, 0.5659109, -0.5490333, 0.5374321
1: 0.0373445, 0.5012081, -0.0585842, 0.5600882, -0.5227437, 0.5597923
2: 0.0585996, 0.4470065, -0.0418383, 0.5510719, -0.4924724, 0.4888448
3: -0.0116420, 0.4992789, -0.0992109, 0.5518441, -0.5634860, 0.5984898
4: 0.0119382, 0.4438587, -0.1008286, 0.5853299, -0.5733917, 0.5446874

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720193, upper bound: 0.5399059
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720193, upper bound: 0.5399059
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0300477, 0.4813079, -0.0510598, 0.5348009, -0.5648485, 0.5323677
1: -0.0013833, 0.5196805, -0.0214454, 0.5362778, -0.5376611, 0.5411259
2: 0.0164753, 0.4788888, -0.0051125, 0.5333415, -0.5168662, 0.4840014
3: -0.0416090, 0.5189773, -0.0692544, 0.5269960, -0.5686050, 0.5882317
4: -0.0368971, 0.4754637, -0.0645348, 0.5658602, -0.6027573, 0.5399985

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709699, upper bound: 0.5543248
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5709699, upper bound: 0.5544301
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0300477, 0.4813079, -0.0825888, 0.5659109, -0.5959586, 0.5638967
1: -0.0013833, 0.5196805, -0.0585842, 0.5600882, -0.5614715, 0.5782647
2: 0.0164753, 0.4788888, -0.0418383, 0.5510719, -0.5345966, 0.5207272
3: -0.0416090, 0.5189773, -0.0992109, 0.5518441, -0.5934531, 0.6181881
4: -0.0368971, 0.4754637, -0.1008286, 0.5853299, -0.6222270, 0.5762923

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710534, upper bound: 0.5543248
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5710534, upper bound: 0.5544301
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0168777, 0.4548433, 0.0168777, 0.4548433, -0.4379657, 0.4379657
1: 0.0373445, 0.5012081, 0.0373445, 0.5012081, -0.4638636, 0.4638636
2: 0.0585996, 0.4470065, 0.0585996, 0.4470065, -0.3884069, 0.3884069
3: -0.0116420, 0.4992789, -0.0116420, 0.4992789, -0.5109209, 0.5109209
4: 0.0119382, 0.4438587, 0.0119382, 0.4438587, -0.4319205, 0.4319205

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5851920, upper bound: 0.5733942
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720391, upper bound: 0.5693455
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0168777, 0.4548433, -0.0300477, 0.4813079, -0.4644302, 0.4848910
1: 0.0373445, 0.5012081, -0.0013833, 0.5196805, -0.4823360, 0.5025914
2: 0.0585996, 0.4470065, 0.0164753, 0.4788888, -0.4202892, 0.4305312
3: -0.0116420, 0.4992789, -0.0416090, 0.5189773, -0.5306193, 0.5408880
4: 0.0119382, 0.4438587, -0.0368971, 0.4754637, -0.4635255, 0.4807558

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5851921, upper bound: 0.5733942
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5720391, upper bound: 0.5693455
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0300477, 0.4813079, 0.0168777, 0.4548433, -0.4848910, 0.4644302
1: -0.0013833, 0.5196805, 0.0373445, 0.5012081, -0.5025914, 0.4823360
2: 0.0164753, 0.4788888, 0.0585996, 0.4470065, -0.4305312, 0.4202892
3: -0.0416090, 0.5189773, -0.0116420, 0.4992789, -0.5408880, 0.5306193
4: -0.0368971, 0.4754637, 0.0119382, 0.4438587, -0.4807558, 0.4635255

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5658967, upper bound: 0.5753921
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5647113, upper bound: 0.5739388
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0300477, 0.4813079, -0.0300477, 0.4813079, -0.5113555, 0.5113555
1: -0.0013833, 0.5196805, -0.0013833, 0.5196805, -0.5210638, 0.5210638
2: 0.0164753, 0.4788888, 0.0164753, 0.4788888, -0.4624135, 0.4624135
3: -0.0416090, 0.5189773, -0.0416090, 0.5189773, -0.5605863, 0.5605863
4: -0.0368971, 0.4754637, -0.0368971, 0.4754637, -0.5123608, 0.5123608

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5658967, upper bound: 0.5753921
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5647113, upper bound: 0.5739388
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0465395, 0.6540892, -0.7051491, 0.5813404
1: -0.0214454, 0.5362778, -0.0209305, 0.6510010, -0.6724464, 0.5572083
2: -0.0051125, 0.5333415, -0.0117567, 0.6184078, -0.6235204, 0.5450982
3: -0.0692544, 0.5269960, -0.0549345, 0.6207125, -0.6899669, 0.5819305
4: -0.0645348, 0.5658602, -0.0604364, 0.6286552, -0.6931900, 0.6262966

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0363288, 0.6410381, -0.6920979, 0.5711297
1: -0.0214454, 0.5362778, -0.0170506, 0.6463324, -0.6677778, 0.5533284
2: -0.0051125, 0.5333415, -0.0088149, 0.6185240, -0.6236365, 0.5421565
3: -0.0692544, 0.5269960, -0.0517277, 0.6142127, -0.6834671, 0.5787236
4: -0.0645348, 0.5658602, -0.0549340, 0.6248241, -0.6893589, 0.6207942

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0465395, 0.6540892, -0.7366780, 0.6124505
1: -0.0585842, 0.5600882, -0.0209305, 0.6510010, -0.7095852, 0.5810187
2: -0.0418383, 0.5510719, -0.0117567, 0.6184078, -0.6602461, 0.5628286
3: -0.0992109, 0.5518441, -0.0549345, 0.6207125, -0.7199233, 0.6067786
4: -0.1008286, 0.5853299, -0.0604364, 0.6286552, -0.7294838, 0.6457663

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0363288, 0.6410381, -0.7236269, 0.6022397
1: -0.0585842, 0.5600882, -0.0170506, 0.6463324, -0.7049166, 0.5771388
2: -0.0418383, 0.5510719, -0.0088149, 0.6185240, -0.6603623, 0.5598869
3: -0.0992109, 0.5518441, -0.0517277, 0.6142127, -0.7134236, 0.6035717
4: -0.1008286, 0.5853299, -0.0549340, 0.6248241, -0.7256528, 0.6402639

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0676533, 0.6910655, -0.7421253, 0.6024542
1: -0.0214454, 0.5362778, -0.0455999, 0.7031308, -0.7245762, 0.5818777
2: -0.0051125, 0.5333415, -0.0339785, 0.6572912, -0.6624038, 0.5673200
3: -0.0692544, 0.5269960, -0.0652640, 0.6937898, -0.7630442, 0.5922599
4: -0.0645348, 0.5658602, -0.0847011, 0.6410571, -0.7055919, 0.6505613

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0510598, 0.5348009, -0.0566970, 0.6792859, -0.7303457, 0.5914979
1: -0.0214454, 0.5362778, -0.0404696, 0.7016609, -0.7231063, 0.5767474
2: -0.0051125, 0.5333415, -0.0304444, 0.6600475, -0.6651601, 0.5637859
3: -0.0692544, 0.5269960, -0.0613841, 0.6990961, -0.7683505, 0.5883800
4: -0.0645348, 0.5658602, -0.0785533, 0.6437433, -0.7082781, 0.6444135

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0676533, 0.6910655, -0.7736543, 0.6335642
1: -0.0585842, 0.5600882, -0.0455999, 0.7031308, -0.7617151, 0.6056881
2: -0.0418383, 0.5510719, -0.0339785, 0.6572912, -0.6991295, 0.5850504
3: -0.0992109, 0.5518441, -0.0652640, 0.6937898, -0.7930007, 0.6171080
4: -0.1008286, 0.5853299, -0.0847011, 0.6410571, -0.7418857, 0.6700310

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0825888, 0.5659109, -0.0566970, 0.6792859, -0.7618747, 0.6226079
1: -0.0585842, 0.5600882, -0.0404696, 0.7016609, -0.7602451, 0.6005578
2: -0.0418383, 0.5510719, -0.0304444, 0.6600475, -0.7018858, 0.5815163
3: -0.0992109, 0.5518441, -0.0613841, 0.6990961, -0.7983069, 0.6132281
4: -0.1008286, 0.5853299, -0.0785533, 0.6437433, -0.7445720, 0.6638832

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0483578, 0.5017393, -0.0465395, 0.6540892, -0.7024471, 0.5482789
1: -0.0165561, 0.5322876, -0.0209305, 0.6510010, -0.6675571, 0.5532181
2: -0.0020304, 0.4945726, -0.0117567, 0.6184078, -0.6204382, 0.5063292
3: -0.0596782, 0.5316396, -0.0549345, 0.6207125, -0.6803907, 0.5865741
4: -0.0589769, 0.5026535, -0.0604364, 0.6286552, -0.6876321, 0.5630898

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0483578, 0.5017393, -0.0363288, 0.6410381, -0.6893959, 0.5380681
1: -0.0165561, 0.5322876, -0.0170506, 0.6463324, -0.6628885, 0.5493382
2: -0.0020304, 0.4945726, -0.0088149, 0.6185240, -0.6205544, 0.5033875
3: -0.0596782, 0.5316396, -0.0517277, 0.6142127, -0.6738909, 0.5833672
4: -0.0589769, 0.5026535, -0.0549340, 0.6248241, -0.6838010, 0.5575875

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0551119, 0.5142183, -0.0465395, 0.6540892, -0.7092011, 0.5607578
1: -0.0310310, 0.5400121, -0.0209305, 0.6510010, -0.6820320, 0.5609426
2: -0.0154766, 0.5017341, -0.0117567, 0.6184078, -0.6338844, 0.5134907
3: -0.0722581, 0.5358996, -0.0549345, 0.6207125, -0.6929706, 0.5908341
4: -0.0708025, 0.5060241, -0.0604364, 0.6286552, -0.6994576, 0.5664604

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0551119, 0.5142183, -0.0363288, 0.6410381, -0.6961499, 0.5505471
1: -0.0310310, 0.5400121, -0.0170506, 0.6463324, -0.6773634, 0.5570627
2: -0.0154766, 0.5017341, -0.0088149, 0.6185240, -0.6340006, 0.5105490
3: -0.0722581, 0.5358996, -0.0517277, 0.6142127, -0.6864708, 0.5876273
4: -0.0708025, 0.5060241, -0.0549340, 0.6248241, -0.6956266, 0.5609581

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0483578, 0.5017393, -0.0676533, 0.6910655, -0.7394233, 0.5693926
1: -0.0165561, 0.5322876, -0.0455999, 0.7031308, -0.7196870, 0.5778875
2: -0.0020304, 0.4945726, -0.0339785, 0.6572912, -0.6593216, 0.5285510
3: -0.0596782, 0.5316396, -0.0652640, 0.6937898, -0.7534680, 0.5969036
4: -0.0589769, 0.5026535, -0.0847011, 0.6410571, -0.7000340, 0.5873545

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0483578, 0.5017393, -0.0566970, 0.6792859, -0.7276437, 0.5584363
1: -0.0165561, 0.5322876, -0.0404696, 0.7016609, -0.7182170, 0.5727572
2: -0.0020304, 0.4945726, -0.0304444, 0.6600475, -0.6620779, 0.5250169
3: -0.0596782, 0.5316396, -0.0613841, 0.6990961, -0.7587743, 0.5930237
4: -0.0589769, 0.5026535, -0.0785533, 0.6437433, -0.7027202, 0.5812068

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0551119, 0.5142183, -0.0676533, 0.6910655, -0.7461774, 0.5818716
1: -0.0310310, 0.5400121, -0.0455999, 0.7031308, -0.7341619, 0.5856121
2: -0.0154766, 0.5017341, -0.0339785, 0.6572912, -0.6727678, 0.5357125
3: -0.0722581, 0.5358996, -0.0652640, 0.6937898, -0.7660480, 0.6011636
4: -0.0708025, 0.5060241, -0.0847011, 0.6410571, -0.7118595, 0.5907251

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0551119, 0.5142183, -0.0566970, 0.6792859, -0.7343978, 0.5709153
1: -0.0310310, 0.5400121, -0.0404696, 0.7016609, -0.7326919, 0.5804818
2: -0.0154766, 0.5017341, -0.0304444, 0.6600475, -0.6755241, 0.5321785
3: -0.0722581, 0.5358996, -0.0613841, 0.6990961, -0.7713542, 0.5972837
4: -0.0708025, 0.5060241, -0.0785533, 0.6437433, -0.7145458, 0.5845774

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0510598, 0.5348009, -0.5813404, 0.7051491
1: -0.0209305, 0.6510010, -0.0214454, 0.5362778, -0.5572083, 0.6724464
2: -0.0117567, 0.6184078, -0.0051125, 0.5333415, -0.5450982, 0.6235204
3: -0.0549345, 0.6207125, -0.0692544, 0.5269960, -0.5819305, 0.6899669
4: -0.0604364, 0.6286552, -0.0645348, 0.5658602, -0.6262966, 0.6931900

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0825888, 0.5659109, -0.6124505, 0.7366780
1: -0.0209305, 0.6510010, -0.0585842, 0.5600882, -0.5810187, 0.7095852
2: -0.0117567, 0.6184078, -0.0418383, 0.5510719, -0.5628286, 0.6602461
3: -0.0549345, 0.6207125, -0.0992109, 0.5518441, -0.6067786, 0.7199233
4: -0.0604364, 0.6286552, -0.1008286, 0.5853299, -0.6457663, 0.7294838

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0363288, 0.6410381, -0.0510598, 0.5348009, -0.5711297, 0.6920979
1: -0.0170506, 0.6463324, -0.0214454, 0.5362778, -0.5533284, 0.6677778
2: -0.0088149, 0.6185240, -0.0051125, 0.5333415, -0.5421565, 0.6236365
3: -0.0517277, 0.6142127, -0.0692544, 0.5269960, -0.5787236, 0.6834671
4: -0.0549340, 0.6248241, -0.0645348, 0.5658602, -0.6207942, 0.6893589

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0363288, 0.6410381, -0.0825888, 0.5659109, -0.6022397, 0.7236269
1: -0.0170506, 0.6463324, -0.0585842, 0.5600882, -0.5771388, 0.7049166
2: -0.0088149, 0.6185240, -0.0418383, 0.5510719, -0.5598869, 0.6603623
3: -0.0517277, 0.6142127, -0.0992109, 0.5518441, -0.6035717, 0.7134236
4: -0.0549340, 0.6248241, -0.1008286, 0.5853299, -0.6402639, 0.7256528

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0483578, 0.5017393, -0.5482789, 0.7024471
1: -0.0209305, 0.6510010, -0.0165561, 0.5322876, -0.5532181, 0.6675571
2: -0.0117567, 0.6184078, -0.0020304, 0.4945726, -0.5063292, 0.6204382
3: -0.0549345, 0.6207125, -0.0596782, 0.5316396, -0.5865741, 0.6803907
4: -0.0604364, 0.6286552, -0.0589769, 0.5026535, -0.5630898, 0.6876321

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0465395, 0.6540892, -0.0551119, 0.5142183, -0.5607578, 0.7092011
1: -0.0209305, 0.6510010, -0.0310310, 0.5400121, -0.5609426, 0.6820320
2: -0.0117567, 0.6184078, -0.0154766, 0.5017341, -0.5134907, 0.6338844
3: -0.0549345, 0.6207125, -0.0722581, 0.5358996, -0.5908341, 0.6929706
4: -0.0604364, 0.6286552, -0.0708025, 0.5060241, -0.5664604, 0.6994576

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0363288, 0.6410381, -0.0483578, 0.5017393, -0.5380681, 0.6893959
1: -0.0170506, 0.6463324, -0.0165561, 0.5322876, -0.5493382, 0.6628885
2: -0.0088149, 0.6185240, -0.0020304, 0.4945726, -0.5033875, 0.6205544
3: -0.0517277, 0.6142127, -0.0596782, 0.5316396, -0.5833672, 0.6738909
4: -0.0549340, 0.6248241, -0.0589769, 0.5026535, -0.5575875, 0.6838010

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0363288, 0.6410381, -0.0551119, 0.5142183, -0.5505471, 0.6961499
1: -0.0170506, 0.6463324, -0.0310310, 0.5400121, -0.5570627, 0.6773634
2: -0.0088149, 0.6185240, -0.0154766, 0.5017341, -0.5105490, 0.6340006
3: -0.0517277, 0.6142127, -0.0722581, 0.5358996, -0.5876273, 0.6864708
4: -0.0549340, 0.6248241, -0.0708025, 0.5060241, -0.5609581, 0.6956266

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0510598, 0.5348009, -0.6024542, 0.7421253
1: -0.0455999, 0.7031308, -0.0214454, 0.5362778, -0.5818777, 0.7245762
2: -0.0339785, 0.6572912, -0.0051125, 0.5333415, -0.5673200, 0.6624038
3: -0.0652640, 0.6937898, -0.0692544, 0.5269960, -0.5922599, 0.7630442
4: -0.0847011, 0.6410571, -0.0645348, 0.5658602, -0.6505613, 0.7055919

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0825888, 0.5659109, -0.6335642, 0.7736543
1: -0.0455999, 0.7031308, -0.0585842, 0.5600882, -0.6056881, 0.7617151
2: -0.0339785, 0.6572912, -0.0418383, 0.5510719, -0.5850504, 0.6991295
3: -0.0652640, 0.6937898, -0.0992109, 0.5518441, -0.6171080, 0.7930007
4: -0.0847011, 0.6410571, -0.1008286, 0.5853299, -0.6700310, 0.7418857

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0566970, 0.6792859, -0.0510598, 0.5348009, -0.5914979, 0.7303457
1: -0.0404696, 0.7016609, -0.0214454, 0.5362778, -0.5767474, 0.7231063
2: -0.0304444, 0.6600475, -0.0051125, 0.5333415, -0.5637859, 0.6651601
3: -0.0613841, 0.6990961, -0.0692544, 0.5269960, -0.5883800, 0.7683505
4: -0.0785533, 0.6437433, -0.0645348, 0.5658602, -0.6444135, 0.7082781

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0566970, 0.6792859, -0.0825888, 0.5659109, -0.6226079, 0.7618747
1: -0.0404696, 0.7016609, -0.0585842, 0.5600882, -0.6005578, 0.7602451
2: -0.0304444, 0.6600475, -0.0418383, 0.5510719, -0.5815163, 0.7018858
3: -0.0613841, 0.6990961, -0.0992109, 0.5518441, -0.6132281, 0.7983069
4: -0.0785533, 0.6437433, -0.1008286, 0.5853299, -0.6638832, 0.7445720

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0483578, 0.5017393, -0.5693926, 0.7394233
1: -0.0455999, 0.7031308, -0.0165561, 0.5322876, -0.5778875, 0.7196870
2: -0.0339785, 0.6572912, -0.0020304, 0.4945726, -0.5285510, 0.6593216
3: -0.0652640, 0.6937898, -0.0596782, 0.5316396, -0.5969036, 0.7534680
4: -0.0847011, 0.6410571, -0.0589769, 0.5026535, -0.5873545, 0.7000340

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0676533, 0.6910655, -0.0551119, 0.5142183, -0.5818716, 0.7461774
1: -0.0455999, 0.7031308, -0.0310310, 0.5400121, -0.5856121, 0.7341619
2: -0.0339785, 0.6572912, -0.0154766, 0.5017341, -0.5357125, 0.6727678
3: -0.0652640, 0.6937898, -0.0722581, 0.5358996, -0.6011636, 0.7660480
4: -0.0847011, 0.6410571, -0.0708025, 0.5060241, -0.5907251, 0.7118595

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0566970, 0.6792859, -0.0483578, 0.5017393, -0.5584363, 0.7276437
1: -0.0404696, 0.7016609, -0.0165561, 0.5322876, -0.5727572, 0.7182170
2: -0.0304444, 0.6600475, -0.0020304, 0.4945726, -0.5250169, 0.6620779
3: -0.0613841, 0.6990961, -0.0596782, 0.5316396, -0.5930237, 0.7587743
4: -0.0785533, 0.6437433, -0.0589769, 0.5026535, -0.5812068, 0.7027202

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0566970, 0.6792859, -0.0551119, 0.5142183, -0.5709153, 0.7343978
1: -0.0404696, 0.7016609, -0.0310310, 0.5400121, -0.5804818, 0.7326919
2: -0.0304444, 0.6600475, -0.0154766, 0.5017341, -0.5321785, 0.6755241
3: -0.0613841, 0.6990961, -0.0722581, 0.5358996, -0.5972837, 0.7713542
4: -0.0785533, 0.6437433, -0.0708025, 0.5060241, -0.5845774, 0.7145458

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.52 + 235.61 = 238.13 seconds
