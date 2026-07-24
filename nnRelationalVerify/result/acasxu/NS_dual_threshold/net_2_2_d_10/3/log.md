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
execution time: IAR + RelationalAnalysis = 1.44 + 0.91 = 2.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5955753, upper bound: 0.5955753

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5855071, upper bound: 0.5927507
time: 0.27 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5878167, upper bound: 0.5878169
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.5855071, upper bound: 0.5927507
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.5878167, upper bound: 0.5878169

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0340946, 0.5122287, -0.0888740, 0.5565974, -0.5906919, 0.6011027
1: -0.0171536, 0.5395310, -0.0582254, 0.5834452, -0.6005988, 0.5977564
2: 0.0010548, 0.5084147, -0.0401547, 0.5380218, -0.5369670, 0.5485694
3: -0.0661722, 0.5292395, -0.0950109, 0.5804102, -0.6465825, 0.6242504
4: -0.0557598, 0.5256444, -0.1025802, 0.5563258, -0.6120856, 0.6282246

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5841669, upper bound: 0.5926599
time: 0.27 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5837792, upper bound: 0.5837792
time: 0.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5837792, upper bound: 0.5878169
time: 0.28 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.1395322, 0.6803006, -0.0888740, 0.5565974, -0.6961296, 0.7691746
1: -0.1182150, 0.6745660, -0.0582254, 0.5834452, -0.7016602, 0.7327914
2: -0.0986016, 0.6600875, -0.0401547, 0.5380218, -0.6366234, 0.7002422
3: -0.1482589, 0.6688396, -0.0950109, 0.5804102, -0.7286692, 0.7638505
4: -0.1795291, 0.7153544, -0.1025802, 0.5563258, -0.7358549, 0.8179346

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5878092, upper bound: 0.5786512
time: 0.28 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5790937, upper bound: 0.5790938
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.67 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.5837792, upper bound: 0.5837792
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.5837792, upper bound: 0.5878169
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.5878092, upper bound: 0.5786512
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.5790937, upper bound: 0.5790938

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0340946, 0.5122287, -0.0340946, 0.5122287, -0.5463233, 0.5463233
1: -0.0171536, 0.5395310, -0.0171536, 0.5395310, -0.5566846, 0.5566846
2: 0.0010548, 0.5084147, 0.0010548, 0.5084147, -0.5073600, 0.5073600
3: -0.0661722, 0.5292395, -0.0661722, 0.5292395, -0.5954117, 0.5954117
4: -0.0557598, 0.5256444, -0.0557598, 0.5256444, -0.5814042, 0.5814042

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5833748, upper bound: 0.5738908
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671131, upper bound: 0.5712816
time: 0.29 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0340946, 0.5122287, -0.1395322, 0.6803006, -0.7143952, 0.6517609
1: -0.0171536, 0.5395310, -0.1182150, 0.6745660, -0.6917197, 0.6577460
2: 0.0010548, 0.5084147, -0.0986016, 0.6600875, -0.6590327, 0.6070163
3: -0.0661722, 0.5292395, -0.1482589, 0.6688396, -0.7350119, 0.6774984
4: -0.0557598, 0.5256444, -0.1795291, 0.7153544, -0.7711142, 0.7051735

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5746556, upper bound: 0.5888277
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5850897, upper bound: 0.5927415
time: 0.32 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.1395322, 0.6803006, -0.0861034, 0.5527080, -0.6922402, 0.7664040
1: -0.1182150, 0.6745660, -0.0552485, 0.5803811, -0.6985961, 0.7298145
2: -0.0986016, 0.6600875, -0.0374962, 0.5354499, -0.6340515, 0.6975837
3: -0.1482589, 0.6688396, -0.0927221, 0.5771117, -0.7253706, 0.7615617
4: -0.1795291, 0.7153544, -0.0997224, 0.5536377, -0.7331668, 0.8150768

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5878092, upper bound: 0.5748528
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5878095, upper bound: 0.5776124
time: 0.29 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.1395322, 0.6803006, -0.1423618, 0.7405249, -0.8800571, 0.8226624
1: -0.1182150, 0.6745660, -0.1086804, 0.7652168, -0.8834317, 0.7832464
2: -0.0986016, 0.6600875, -0.0874307, 0.7105821, -0.8091837, 0.7475182
3: -0.1482589, 0.6688396, -0.1342866, 0.7839161, -0.9321750, 0.8031263
4: -0.1795291, 0.7153544, -0.1427265, 0.6830772, -0.8626062, 0.8580809

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5786510, upper bound: 0.5790939
time: 0.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5786510, upper bound: 0.5790939
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.93 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5833748, upper bound: 0.5738908
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5671131, upper bound: 0.5712816
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5746556, upper bound: 0.5888277
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5850897, upper bound: 0.5927415
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5878092, upper bound: 0.5748528
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5878095, upper bound: 0.5776124
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5786510, upper bound: 0.5790939
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.5786510, upper bound: 0.5790939

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0282927, 0.5038331, -0.0340946, 0.5122287, -0.5405214, 0.5379276
1: -0.0107801, 0.5356642, -0.0171536, 0.5395310, -0.5503111, 0.5528178
2: 0.0066494, 0.5028979, 0.0010548, 0.5084147, -0.5017654, 0.5018431
3: -0.0605013, 0.5245377, -0.0661722, 0.5292395, -0.5897408, 0.5907100
4: -0.0493516, 0.5178238, -0.0557598, 0.5256444, -0.5749960, 0.5735836

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5815302, upper bound: 0.5712951
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5871260, upper bound: 0.5738908
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0497389, 0.5260671, -0.0340946, 0.5122287, -0.5619676, 0.5601617
1: -0.0380906, 0.5548710, -0.0171536, 0.5395310, -0.5776216, 0.5720246
2: -0.0206398, 0.5191786, 0.0010548, 0.5084147, -0.5290545, 0.5181238
3: -0.0844864, 0.5445794, -0.0661722, 0.5292395, -0.6137258, 0.6107517
4: -0.0744687, 0.5306695, -0.0557598, 0.5256444, -0.6001132, 0.5864293

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608442, upper bound: 0.5680552
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608442, upper bound: 0.5712816
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0313344, 0.5309770, -0.1395322, 0.6803006, -0.7116350, 0.6705092
1: -0.0176833, 0.5304160, -0.1182150, 0.6745660, -0.6922493, 0.6486310
2: 0.0016118, 0.5352706, -0.0986016, 0.6600875, -0.6584756, 0.6338722
3: -0.0702939, 0.5169436, -0.1482589, 0.6688396, -0.7391335, 0.6652026
4: -0.0548029, 0.5717695, -0.1795291, 0.7153544, -0.7701573, 0.7512985

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5552424, upper bound: 0.5854336
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0249293, 0.5028042, -0.1395322, 0.6803006, -0.7052299, 0.6423364
1: -0.0083491, 0.5293792, -0.1182150, 0.6745660, -0.6829151, 0.6475942
2: 0.0091363, 0.5020263, -0.0986016, 0.6600875, -0.6509511, 0.6006278
3: -0.0586933, 0.5183036, -0.1482589, 0.6688396, -0.7275329, 0.6665625
4: -0.0479572, 0.5192338, -0.1795291, 0.7153544, -0.7633116, 0.6987629

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5762055, upper bound: 0.5927338
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5690616, upper bound: 0.5898438
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671131, upper bound: 0.5736607
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1395322, 0.6803006, -0.0314338, 0.5096861, -0.6492183, 0.7117344
1: -0.1182150, 0.6745660, -0.0142598, 0.5382812, -0.6564962, 0.6888258
2: -0.0986016, 0.6600875, 0.0035928, 0.5071027, -0.6057043, 0.6564946
3: -0.1482589, 0.6688396, -0.0640581, 0.5277065, -0.6759654, 0.7328977
4: -0.1795291, 0.7153544, -0.0529619, 0.5238895, -0.7034186, 0.7683163

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5873665, upper bound: 0.5748528
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5873665, upper bound: 0.5748528
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1395322, 0.6803006, -0.1370877, 0.6449429, -0.7844751, 0.8173883
1: -0.1182150, 0.6745660, -0.1157942, 0.6481835, -0.7663984, 0.7903602
2: -0.0986016, 0.6600875, -0.0963328, 0.6399269, -0.7385285, 0.7564203
3: -0.1482589, 0.6688396, -0.1462929, 0.6376438, -0.7859027, 0.8151326
4: -0.1795291, 0.7153544, -0.1769826, 0.7004614, -0.8799905, 0.8923370

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5873667, upper bound: 0.5776124
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5873667, upper bound: 0.5776124
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1370877, 0.6742949, -0.1423618, 0.7405249, -0.8776126, 0.8166567
1: -0.1157942, 0.6703373, -0.1086804, 0.7652168, -0.8810109, 0.7790177
2: -0.0963328, 0.6555887, -0.0874307, 0.7105821, -0.8069149, 0.7430195
3: -0.1462929, 0.6638429, -0.1342866, 0.7839161, -0.9302090, 0.7981296
4: -0.1769826, 0.7124667, -0.1427265, 0.6830772, -0.8600598, 0.8551932

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1624700, 0.7454740, -0.1423618, 0.7405249, -0.9029949, 0.8878357
1: -0.1339076, 0.7574279, -0.1086804, 0.7652168, -0.8991243, 0.8661083
2: -0.1090510, 0.7119780, -0.0874307, 0.7105821, -0.8196331, 0.7994087
3: -0.1604708, 0.7595687, -0.1342866, 0.7839161, -0.9443869, 0.8938553
4: -0.1788951, 0.7198949, -0.1427265, 0.6830772, -0.8619723, 0.8626214

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.35 + 32.84 = 35.19 seconds
