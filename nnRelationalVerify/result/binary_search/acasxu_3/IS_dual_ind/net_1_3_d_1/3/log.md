## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 1.91709821175


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340)
1: (-0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480)
2: (-1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007)
3: (-1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638)
4: (-2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280)

## BASE Result
execution time: IAR + LP analysis = 1.41 + 1.13 = 2.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.9315851, upper bound: 1.9315851


# Binary Search by BASE starts (time budget: 1197.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=2.269134044647217
rel_dist={0: [-1.9301230654393355, 1.9301230654393358]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=2.269134044647217
rel_dist={0: [-1.927168397752424, 1.9271683977524248]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0113636, mid=0.0113636, abs_max=2.269134044647217
rel_dist={0: [-1.9245229523020178, 1.9245229523020182]}

## Binary search (step 4) starts
Candidate diff: 0.0056818


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0056818, mid=0.0056818, abs_max=2.269134044647217
rel_dist={0: [-1.9223012200559109, 1.9223012200559104]}

## Binary search (step 5) starts
Candidate diff: 0.0028409


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0028409, mid=0.0028409, abs_max=2.269134044647217
rel_dist={0: [-1.92097236095474, 1.9209723609547407]}

## Binary search (step 6) starts
Candidate diff: 0.0014205


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0014205, mid=0.0014205, abs_max=2.269134044647217
rel_dist={0: [-1.9197152010908964, 1.919715201090896]}

## Binary search (step 7) starts
Candidate diff: 0.0007102


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007102, mid=0.0007102, abs_max=2.269134044647217
rel_dist={0: [-1.918877523994997, 1.918877523994997]}

## Binary search (step 8) starts
Candidate diff: 0.0003551


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003551, mid=0.0003551, abs_max=2.269134044647217
rel_dist={0: [-1.9184175972727442, 1.9184175972725122]}

## Binary search (step 9) starts
Candidate diff: 0.0001776


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001776, mid=0.0001776, abs_max=2.269134044647217
rel_dist={0: [-1.9181875428014488, 1.9181875428014488]}

## Binary search (step 10) starts
Candidate diff: 0.0000888


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000888, mid=0.0000888, abs_max=2.269134044647217
rel_dist={0: [-1.9180725052024918, 1.918072505202593]}

## Binary search (step 11) starts
Candidate diff: 0.0000444


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000444, mid=0.0000444, abs_max=2.269134044647217
rel_dist={0: [-1.9180149864222327, 1.9180149864223193]}

## Binary search (step 12) starts
Candidate diff: 0.0000222


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000222, mid=0.0000222, abs_max=2.269134044647217
rel_dist={0: [-1.9179862270484023, 1.9179862270483863]}

## Binary search (step 13) starts
Candidate diff: 0.0000111


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000111, mid=0.0000111, abs_max=2.269134044647217
rel_dist={0: [-1.9179717717444311, 1.917971771744413]}

## Binary search (step 14) starts
Candidate diff: 0.0000055


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000055, mid=0.0000055, abs_max=2.269134044647217
rel_dist={0: [-1.9179640661799573, 1.9179640661799482]}

## Binary search (step 15) starts
Candidate diff: 0.0000028


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000028, mid=0.0000028, abs_max=2.269134044647217
rel_dist={0: [-1.9179602134166283, 1.9179602134166238]}

## Binary search (step 16) starts
Candidate diff: 0.0000014


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000014, mid=0.0000014, abs_max=2.269134044647217
rel_dist={0: [-1.9179582870654106, 1.9179582870654137]}

## Binary search (step 17) starts
Candidate diff: 0.0000007


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000007, mid=0.0000007, abs_max=2.269134044647217
rel_dist={0: [-1.9179573473512748, 1.9179573258473672]}

## Binary Search Result
Binary search time: 44.69 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1152.77 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9262004, upper bound: 1.9232330
time: 0.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9314710, upper bound: 1.9314708
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -1.9262004, upper bound: 1.9232330
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -1.9314710, upper bound: 1.9314708

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.6770015, 1.4656314, -2.0459304, 2.1212513
1: -0.6313032, 2.1405544, -0.7475953, 2.2259817, -2.8572850, 2.8881497
2: -1.3848062, 1.5734707, -1.6076193, 1.5682487, -2.9530549, 3.1810899
3: -0.9428792, 3.0088511, -1.0890913, 3.4468250, -4.3897042, 4.0979424
4: -1.8759956, 1.7349151, -2.1315250, 1.7607409, -3.6367364, 3.8664403

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9188660, upper bound: 1.9188660
time: 0.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9188660, upper bound: 1.9231637
time: 0.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7438335, 1.5253004, -2.2143774, 2.2126141
1: -0.7561793, 2.1721611, -0.8104258, 2.3001223, -3.0563016, 2.9825869
2: -1.6216621, 1.5724510, -1.7379694, 1.6217314, -3.2433934, 3.3104205
3: -1.0972075, 3.4126759, -1.1569667, 3.6523972, -4.7496047, 4.5696425
4: -2.1124964, 1.7667136, -2.2834320, 1.8319958, -3.9444923, 4.0501456

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9231637, upper bound: 1.9262004
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9231637, upper bound: 1.9314710
time: 0.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.9188660, upper bound: 1.9188660
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.9188660, upper bound: 1.9231637
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.9231637, upper bound: 1.9262004
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -1.9231637, upper bound: 1.9314710

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.5802990, 1.4442499, -2.0245490, 2.0245490
1: -0.6313032, 2.1405544, -0.6313032, 2.1405544, -2.7718577, 2.7718577
2: -1.3848062, 1.5734707, -1.3848062, 1.5734707, -2.9582767, 2.9582767
3: -0.9428792, 3.0088511, -0.9428792, 3.0088511, -3.9517303, 3.9517303
4: -1.8759956, 1.7349151, -1.8759956, 1.7349151, -3.6109109, 3.6109109

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9183221
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9169984
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.6890769, 1.4687806, -2.0490797, 2.1333268
1: -0.6313032, 2.1405544, -0.7561793, 2.1721611, -2.8034644, 2.8967338
2: -1.3848062, 1.5734707, -1.6216621, 1.5724510, -2.9572573, 3.1951327
3: -0.9428792, 3.0088511, -1.0972075, 3.4126759, -4.3555551, 4.1060586
4: -1.8759956, 1.7349151, -2.1124964, 1.7667136, -3.6427093, 3.8474116

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.5802990, 1.4442499, -2.1333268, 2.0490797
1: -0.7561793, 2.1721611, -0.6313032, 2.1405544, -2.8967338, 2.8034644
2: -1.6216621, 1.5724510, -1.3848062, 1.5734707, -3.1951327, 2.9572573
3: -1.0972075, 3.4126759, -0.9428792, 3.0088511, -4.1060586, 4.3555551
4: -2.1124964, 1.7667136, -1.8759956, 1.7349151, -3.8474116, 3.6427093

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9256805
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9157402
time: 0.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.6890769, 1.4687806, -2.1578574, 2.1578574
1: -0.7561793, 2.1721611, -0.7561793, 2.1721611, -2.9283404, 2.9283404
2: -1.6216621, 1.5724510, -1.6216621, 1.5724510, -3.1941133, 3.1941133
3: -1.0972075, 3.4126759, -1.0972075, 3.4126759, -4.5098834, 4.5098834
4: -2.1124964, 1.7667136, -2.1124964, 1.7667136, -3.8792100, 3.8792100

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9304277
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9183079
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9183221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9169984
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9256805
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9157402
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9304277
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9183079

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.5802990, 1.4442499, -1.9805572, 1.9731890
1: -0.5818403, 2.0295284, -0.6313032, 2.1405544, -2.7223947, 2.6608315
2: -1.2697520, 1.5284014, -1.3848062, 1.5734707, -2.8432226, 2.9132075
3: -0.8890121, 2.7896709, -0.9428792, 3.0088511, -3.8978631, 3.7325501
4: -1.7242160, 1.6678995, -1.8759956, 1.7349151, -3.4591312, 3.5438952

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9111746
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9169984
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6890769, 1.4687806, -2.0050879, 2.0819669
1: -0.5818403, 2.0295284, -0.7561793, 2.1721611, -2.7540014, 2.7857077
2: -1.2697520, 1.5284014, -1.6216621, 1.5724510, -2.8422031, 3.1500635
3: -0.8890121, 2.7896709, -1.0972075, 3.4126759, -4.3016882, 3.8868785
4: -1.7242160, 1.6678995, -2.1124964, 1.7667136, -3.4909296, 3.7803960

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6890769, 1.4687806, -2.0975733, 2.1717911
1: -0.6841701, 2.1662390, -0.7561793, 2.1721611, -2.8563313, 2.9224184
2: -1.4798999, 1.6526315, -1.6216621, 1.5724510, -3.0523510, 3.2742937
3: -0.9925163, 3.1035147, -1.0972075, 3.4126759, -4.4051924, 4.2007222
4: -1.9907460, 1.8352203, -2.1124964, 1.7667136, -3.7574596, 3.9477167

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5802990, 1.4442499, -2.0735838, 1.9906434
1: -0.6947789, 2.0680566, -0.6313032, 2.1405544, -2.8353333, 2.6993599
2: -1.4880743, 1.5177747, -1.3848062, 1.5734707, -3.0615449, 2.9025807
3: -1.0287070, 3.1677570, -0.9428792, 3.0088511, -4.0375581, 4.1106362
4: -1.9440632, 1.6937697, -1.8759956, 1.7349151, -3.6789784, 3.5697653

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9164596, upper bound: 1.9250301
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5802990, 1.4442499, -2.1661160, 2.0873237
1: -0.7863536, 2.1740503, -0.6313032, 2.1405544, -2.9269080, 2.8053536
2: -1.6623611, 1.6327487, -1.3848062, 1.5734707, -3.2358317, 3.0175548
3: -1.1238203, 3.4061260, -0.9428792, 3.0088511, -4.1326714, 4.3490052
4: -2.1544065, 1.8428237, -1.8759956, 1.7349151, -3.8893218, 3.7188194

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6890769, 1.4687806, -2.0981145, 2.0994213
1: -0.6947789, 2.0680566, -0.7561793, 2.1721611, -2.8669400, 2.8242359
2: -1.4880743, 1.5177747, -1.6216621, 1.5724510, -3.0605254, 3.1394367
3: -1.0287070, 3.1677570, -1.0972075, 3.4126759, -4.4413829, 4.2649646
4: -1.9440632, 1.6937697, -2.1124964, 1.7667136, -3.7107768, 3.8062661

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277942
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6890769, 1.4687806, -2.1906466, 2.1961017
1: -0.7863536, 2.1740503, -0.7561793, 2.1721611, -2.9585147, 2.9302297
2: -1.6623611, 1.6327487, -1.6216621, 1.5724510, -3.2348123, 3.2544107
3: -1.1238203, 3.4061260, -1.0972075, 3.4126759, -4.5364962, 4.5033336
4: -2.1544065, 1.8428237, -2.1124964, 1.7667136, -3.9211202, 3.9553201

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9164305, upper bound: 1.9153593
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.33 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9111746
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9169984
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277942
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9164305, upper bound: 1.9153593
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9256805
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6235747, 1.4036301, -0.5006113, 1.2452075, -1.8687823, 1.9042414
1: -0.6893435, 2.0595088, -0.6059399, 2.0574260, -2.7467694, 2.6654487
2: -1.4771109, 1.5110943, -1.3041267, 1.3373374, -2.8144484, 2.8152211
3: -1.0233059, 3.1524611, -0.9207745, 3.0303936, -4.0536995, 4.0732355
4: -1.9305515, 1.6855165, -1.7729893, 1.4975882, -3.4281397, 3.4585056

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9164293, upper bound: 1.9153593
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6191602, 1.3756136, -2.0049477, 2.0295045
1: -0.6947789, 2.0680566, -0.6878209, 2.0247331, -2.7195120, 2.7558775
2: -1.4880743, 1.5177747, -1.4858418, 1.4780548, -2.9661291, 3.0036163
3: -1.0287070, 3.1677570, -1.0179358, 3.1737194, -4.2024264, 4.1856928
4: -1.9440632, 1.6937697, -1.9349661, 1.6475326, -3.5915956, 3.6287358

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031
time: 0.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.39 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9256805
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9164293, upper bound: 1.9153593
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.39
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9256806, upper bound: 1.9196396
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9220986, upper bound: 1.9099164
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5006113, 1.2452075, -1.8745415, 1.9109557
1: -0.6947789, 2.0680566, -0.6059399, 2.0574260, -2.7522049, 2.6739964
2: -1.4880743, 1.5177747, -1.3041267, 1.3373374, -2.8254118, 2.8219013
3: -1.0287070, 3.1677570, -0.9207745, 3.0303936, -4.0591006, 4.0885315
4: -1.9440632, 1.6937697, -1.7729893, 1.4975882, -3.4416513, 3.4667590

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6191602, 1.3756136, -2.0049477, 2.0295045
1: -0.6947789, 2.0680566, -0.6878209, 2.0247331, -2.7195120, 2.7558775
2: -1.4880743, 1.5177747, -1.4858418, 1.4780548, -2.9661291, 3.0036163
3: -1.0287070, 3.1677570, -1.0179358, 3.1737194, -4.2024264, 4.1856928
4: -1.9440632, 1.6937697, -1.9349661, 1.6475326, -3.5915956, 3.6287358

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.46 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9256806, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9220986, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.46
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9256802
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9256802
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6235747, 1.4036301, -0.5006113, 1.2452075, -1.8687823, 1.9042414
1: -0.6893435, 2.0595088, -0.6059399, 2.0574260, -2.7467694, 2.6654487
2: -1.4771109, 1.5110943, -1.3041267, 1.3373374, -2.8144484, 2.8152211
3: -1.0233059, 3.1524611, -0.9207745, 3.0303936, -4.0536995, 4.0732355
4: -1.9305515, 1.6855165, -1.7729893, 1.4975882, -3.4281397, 3.4585056

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9164293, upper bound: 1.9153593
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5502033, 1.2859328, -1.9152668, 1.9605477
1: -0.6947789, 2.0680566, -0.6213336, 1.9295921, -2.6243711, 2.6893902
2: -1.4880743, 1.5177747, -1.3427572, 1.4046347, -2.8927090, 2.8605318
3: -1.0287070, 3.1677570, -0.9433498, 2.9237518, -3.9524589, 4.1111069
4: -1.9440632, 1.6937697, -1.7625704, 1.5530781, -3.4971414, 3.4563401

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6235747, 1.4036301, -0.4847693, 1.2275558, -1.8511305, 1.8883995
1: -0.6893435, 2.0595088, -0.5906477, 2.0254970, -2.7148404, 2.6501565
2: -1.4771109, 1.5110943, -1.2678347, 1.3213601, -2.7984710, 2.7789290
3: -1.0233059, 3.1524611, -0.9001975, 2.9715080, -3.9948139, 4.0526586
4: -1.9305515, 1.6855165, -1.7272196, 1.4779277, -3.4084792, 3.4127359

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9164293, upper bound: 1.9153593
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6191602, 1.3756136, -2.0049477, 2.0295045
1: -0.6947789, 2.0680566, -0.6878209, 2.0247331, -2.7195120, 2.7558775
2: -1.4880743, 1.5177747, -1.4858418, 1.4780548, -2.9661291, 3.0036163
3: -1.0287070, 3.1677570, -1.0179358, 3.1737194, -4.2024264, 4.1856928
4: -1.9440632, 1.6937697, -1.9349661, 1.6475326, -3.5915956, 3.6287358

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.60 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207749
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9256802
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9198674
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9256802
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9164293, upper bound: 1.9153593
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9164293, upper bound: 1.9153593
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9099031

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9256806, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9256806, upper bound: 1.9196396
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5070248, -2.0433321, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1740503, -2.7558906, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6327487, -2.9025006, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4061260, -4.2951384, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8428237, -3.5670397, 3.8223062

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5070248, -2.1358173, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1740503, -2.8582206, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6327487, -3.1126485, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4061260, -4.3986425, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8428237, -3.8335698, 3.9896269

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9220986, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183955, upper bound: 1.9082926
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9220986, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5006113, 1.2452075, -1.8745415, 1.9109557
1: -0.6947789, 2.0680566, -0.6059399, 2.0574260, -2.7522049, 2.6739964
2: -1.4880743, 1.5177747, -1.3041267, 1.3373374, -2.8254118, 2.8219013
3: -1.0287070, 3.1677570, -0.9207745, 3.0303936, -4.0591006, 4.0885315
4: -1.9440632, 1.6937697, -1.7729893, 1.4975882, -3.4416513, 3.4667590

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5502033, 1.2859328, -1.9152668, 1.9605477
1: -0.6947789, 2.0680566, -0.6213336, 1.9295921, -2.6243711, 2.6893902
2: -1.4880743, 1.5177747, -1.3427572, 1.4046347, -2.8927090, 2.8605318
3: -1.0287070, 3.1677570, -0.9433498, 2.9237518, -3.9524589, 4.1111069
4: -1.9440632, 1.6937697, -1.7625704, 1.5530781, -3.4971414, 3.4563401

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.4847693, 1.2275558, -1.8568897, 1.8951137
1: -0.6947789, 2.0680566, -0.5906477, 2.0254970, -2.7202759, 2.6587043
2: -1.4880743, 1.5177747, -1.2678347, 1.3213601, -2.8094344, 2.7856092
3: -1.0287070, 3.1677570, -0.9001975, 2.9715080, -4.0002151, 4.0679545
4: -1.9440632, 1.6937697, -1.7272196, 1.4779277, -3.4219909, 3.4209893

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6191602, 1.3756136, -2.0049477, 2.0295045
1: -0.6947789, 2.0680566, -0.6878209, 2.0247331, -2.7195120, 2.7558775
2: -1.4880743, 1.5177747, -1.4858418, 1.4780548, -2.9661291, 3.0036163
3: -1.0287070, 3.1677570, -1.0179358, 3.1737194, -4.2024264, 4.1856928
4: -1.9440632, 1.6937697, -1.9349661, 1.6475326, -3.5915956, 3.6287358

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943
time: 0.44 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.95 seconds
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9256806, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9198674, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9256806, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9207748
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9196396
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9220986, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9209672, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9220986, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9196396, upper bound: 1.9157402
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9207749, upper bound: 1.9157402
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9166133, upper bound: 1.9269233
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9241591
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.95
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9277943

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6293340, 1.4103444, -1.9466517, 2.0222239
1: -0.5818403, 2.0295284, -0.6947789, 2.0680566, -2.6498969, 2.7243073
2: -1.2697520, 1.5284014, -1.4880743, 1.5177747, -2.7875266, 3.0164757
3: -0.8890121, 2.7896709, -1.0287070, 3.1677570, -4.0567694, 3.8183780
4: -1.7242160, 1.6678995, -1.9440632, 1.6937697, -3.4179857, 3.6119628

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6293340, 1.4103444, -2.0391369, 2.1120481
1: -0.6841701, 2.1662390, -0.6947789, 2.0680566, -2.7522268, 2.8610179
2: -1.4798999, 1.6526315, -1.4880743, 1.5177747, -2.9976745, 3.1407058
3: -0.9925163, 3.1035147, -1.0287070, 3.1677570, -4.1602736, 4.1322217
4: -1.9907460, 1.8352203, -1.9440632, 1.6937697, -3.6845157, 3.7792835

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.7218661, 1.5069630, -2.1357555, 2.2045803
1: -0.6841701, 2.1662390, -0.7863536, 2.1736212, -2.8577914, 2.9525926
2: -1.4798999, 1.6526315, -1.6623611, 1.6324198, -3.1123197, 3.3149927
3: -0.9925163, 3.1035147, -1.1238203, 3.4058056, -4.3983221, 4.2273350
4: -1.9907460, 1.8352203, -2.1544065, 1.8426332, -3.8333793, 3.9896269

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6290464, 1.4099917, -1.9462991, 2.0219364
1: -0.5818403, 2.0295284, -0.6945271, 2.0676394, -2.6494796, 2.7240555
2: -1.2697520, 1.5284014, -1.4875946, 1.5174351, -2.7871871, 3.0159960
3: -0.8890121, 2.7896709, -1.0283990, 3.1669998, -4.0560122, 3.8180699
4: -1.7242160, 1.6678995, -1.9434557, 1.6933250, -3.4175410, 3.6113553

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.7218661, 1.5069630, -2.0432703, 2.1147561
1: -0.5818403, 2.0295284, -0.7863536, 2.1736212, -2.7554615, 2.8158820
2: -1.2697520, 1.5284014, -1.6623611, 1.6324198, -2.9021719, 3.1907625
3: -0.8890121, 2.7896709, -1.1238203, 3.4058056, -4.2948179, 3.9134912
4: -1.7242160, 1.6678995, -2.1544065, 1.8426332, -3.5668492, 3.8223062

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082926, upper bound: 1.9183955
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9209672
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9196396
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6290464, 1.4099917, -2.0387843, 2.1117606
1: -0.6841701, 2.1662390, -0.6945271, 2.0676394, -2.7518096, 2.8607662
2: -1.4798999, 1.6526315, -1.4875946, 1.5174351, -2.9973350, 3.1402261
3: -0.9925163, 3.1035147, -1.0283990, 3.1669998, -4.1595163, 4.1319137
4: -1.9907460, 1.8352203, -1.9434557, 1.6933250, -3.6840711, 3.7786760

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9220338, upper bound: 1.9219473
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9220338, upper bound: 1.9297754
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.9220338, upper bound: 1.9219473
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.9220338, upper bound: 1.9297754

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.5925088, 1.3923821, -1.9726812, 2.0367587
1: -0.6313032, 2.1405544, -0.6679277, 2.1326437, -2.7639470, 2.8084822
2: -1.3848062, 1.5734707, -1.4484310, 1.4988259, -2.8836322, 3.0219016
3: -0.9428792, 3.0088511, -1.0020442, 3.1850576, -4.1279368, 4.0108953
4: -1.8759956, 1.7349151, -1.9370098, 1.6677970, -3.5437927, 3.6719251

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179842, upper bound: 1.9179842
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179842, upper bound: 1.9219473
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7438335, 1.5253004, -2.2143774, 2.2126141
1: -0.7561793, 2.1721611, -0.8104258, 2.3001223, -3.0563016, 2.9825869
2: -1.6216621, 1.5724510, -1.7379694, 1.6217314, -3.2433934, 3.3104205
3: -1.0972075, 3.4126759, -1.1569667, 3.6523972, -4.7496047, 4.5696425
4: -2.1124964, 1.7667136, -2.2834320, 1.8319958, -3.9444923, 4.0501456

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9219473, upper bound: 1.9220338
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9219473, upper bound: 1.9297755
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.12 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -1.9179842, upper bound: 1.9179842
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -1.9179842, upper bound: 1.9219473
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -1.9219473, upper bound: 1.9220338
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -1.9219473, upper bound: 1.9297755

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.5316627, 1.4084746, -1.9887736, 1.9759126
1: -0.6313032, 2.1405544, -0.5854633, 2.0988042, -2.7301073, 2.7260177
2: -1.3848062, 1.5734707, -1.3146629, 1.5348946, -2.9197006, 2.8881335
3: -0.9428792, 3.0088511, -0.9056726, 2.9524984, -3.8953776, 3.9145236
4: -1.8759956, 1.7349151, -1.7896314, 1.6874362, -3.5634317, 3.5245466

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9167569
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9165404
time: 0.33 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.6778708, 1.4567192, -2.0370183, 2.1221206
1: -0.6313032, 2.1405544, -0.7454648, 2.1587396, -2.7900429, 2.8860192
2: -1.3848062, 1.5734707, -1.6062489, 1.5596941, -2.9445002, 3.1797194
3: -0.9428792, 3.0088511, -1.0908279, 3.3998718, -4.3427510, 4.0996790
4: -1.8759956, 1.7349151, -2.0957193, 1.7520957, -3.6280913, 3.8306346

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.5802990, 1.4442499, -2.1333268, 2.0490797
1: -0.7561793, 2.1721611, -0.6313032, 2.1405544, -2.8967338, 2.8034644
2: -1.6216621, 1.5724510, -1.3848062, 1.5734707, -3.1951327, 2.9572573
3: -1.0972075, 3.4126759, -0.9428792, 3.0088511, -4.1060586, 4.3555551
4: -2.1124964, 1.7667136, -1.8759956, 1.7349151, -3.8474116, 3.6427093

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179293, upper bound: 1.9209028
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9156888
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.6890769, 1.4687806, -2.1578574, 2.1578574
1: -0.7561793, 2.1721611, -0.7561793, 2.1721611, -2.9283404, 2.9283404
2: -1.6216621, 1.5724510, -1.6216621, 1.5724510, -3.1941133, 3.1941133
3: -1.0972075, 3.4126759, -1.0972075, 3.4126759, -4.5098834, 4.5098834
4: -2.1124964, 1.7667136, -2.1124964, 1.7667136, -3.8792100, 3.8792100

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179293, upper bound: 1.9282064
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9167569
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9165404
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9179293, upper bound: 1.9209028
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9156888
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9179293, upper bound: 1.9282064
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6778708, 1.4567192, -1.9930265, 2.0707607
1: -0.5818403, 2.0295284, -0.7454648, 2.1587396, -2.7405798, 2.7749932
2: -1.2697520, 1.5284014, -1.6062489, 1.5596941, -2.8294461, 3.1346502
3: -0.8890121, 2.7896709, -1.0908279, 3.3998718, -4.2888842, 3.8804989
4: -1.7242160, 1.6678995, -2.0957193, 1.7520957, -3.4763117, 3.7636189

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6778708, 1.4567192, -2.0855117, 2.1605849
1: -0.6841701, 2.1662390, -0.7454648, 2.1587396, -2.8429098, 2.9117038
2: -1.4798999, 1.6526315, -1.6062489, 1.5596941, -3.0395939, 3.2588804
3: -0.9925163, 3.1035147, -1.0908279, 3.3998718, -4.3923883, 4.1943426
4: -1.9907460, 1.8352203, -2.0957193, 1.7520957, -3.7428417, 3.9309397

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5802990, 1.4442499, -2.0735838, 1.9906434
1: -0.6947789, 2.0680566, -0.6313032, 2.1405544, -2.8353333, 2.6993599
2: -1.4880743, 1.5177747, -1.3848062, 1.5734707, -3.0615449, 2.9025807
3: -1.0287070, 3.1677570, -0.9428792, 3.0088511, -4.0375581, 4.1106362
4: -1.9440632, 1.6937697, -1.8759956, 1.7349151, -3.6789784, 3.5697653

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.8863969, upper bound: 1.9045974
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9156888
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5802990, 1.4442499, -2.1661160, 2.0873237
1: -0.7863536, 2.1740503, -0.6313032, 2.1405544, -2.9269080, 2.8053536
2: -1.6623611, 1.6327487, -1.3848062, 1.5734707, -3.2358317, 3.0175548
3: -1.1238203, 3.4061260, -0.9428792, 3.0088511, -4.1326714, 4.3490052
4: -2.1544065, 1.8428237, -1.8759956, 1.7349151, -3.8893218, 3.7188194

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9165404, upper bound: 1.9099164
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9165404, upper bound: 1.9156888
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6890769, 1.4687806, -2.0981145, 2.0994213
1: -0.6947789, 2.0680566, -0.7561793, 2.1721611, -2.8669400, 2.8242359
2: -1.4880743, 1.5177747, -1.6216621, 1.5724510, -3.0605254, 3.1394367
3: -1.0287070, 3.1677570, -1.0972075, 3.4126759, -4.4413829, 4.2649646
4: -1.9440632, 1.6937697, -2.1124964, 1.7667136, -3.7107768, 3.8062661

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251726
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6778517, 1.4561149, -2.1779809, 2.1848764
1: -0.7863536, 2.1740503, -0.7446628, 2.1508493, -2.9372029, 2.9187131
2: -1.6623611, 1.6327487, -1.5972052, 1.5597439, -3.2221050, 3.2299538
3: -1.1238203, 3.4061260, -1.0843320, 3.3706455, -4.4944658, 4.4904580
4: -2.1544065, 1.8428237, -2.0810394, 1.7501799, -3.9045863, 3.9238632

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161977
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183080
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.87 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9156888
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9165404, upper bound: 1.9099164
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9165404, upper bound: 1.9156888
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251726
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161977
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183080

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169196
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9209028
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9156888
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5405147, 1.3015049, -0.5006113, 1.2452075, -1.7857223, 1.8021162
1: -0.6153402, 1.9278731, -0.6059399, 2.0574260, -2.6727662, 2.5338130
2: -1.3082781, 1.4133801, -1.3041267, 1.3373374, -2.6456156, 2.7175069
3: -0.9375644, 2.8940592, -0.9207745, 3.0303936, -3.9679580, 3.8148336
4: -1.7347097, 1.5621660, -1.7729893, 1.4975882, -3.2322979, 3.3351552

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9233022
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6181140, 1.3958161, -0.6191602, 1.3756136, -1.9937276, 2.0149763
1: -0.6849847, 2.0469999, -0.6878209, 2.0247331, -2.7097178, 2.7348208
2: -1.4684038, 1.5033507, -1.4858418, 1.4780548, -2.9464586, 2.9891925
3: -1.0164871, 3.1362658, -1.0179358, 3.1737194, -4.1902065, 4.1542015
4: -1.9182425, 1.6757677, -1.9349661, 1.6475326, -3.5657749, 3.6107337

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9149989, upper bound: 1.9099031
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6293340, 1.4103444, -2.1322105, 2.1363587
1: -0.7863536, 2.1740503, -0.6947789, 2.0680566, -2.8544102, 2.8688293
2: -1.6623611, 1.6327487, -1.4880743, 1.5177747, -3.1801357, 3.1208229
3: -1.1238203, 3.4061260, -1.0287070, 3.1677570, -4.2915773, 4.4348330
4: -2.1544065, 1.8428237, -1.9440632, 1.6937697, -3.8481762, 3.7868869

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.7218661, 1.5070248, -2.2288909, 2.2288909
1: -0.7863536, 2.1740503, -0.7863536, 2.1740503, -2.9604039, 2.9604039
2: -1.6623611, 1.6327487, -1.6623611, 1.6327487, -3.2951097, 3.2951097
3: -1.1238203, 3.4061260, -1.1238203, 3.4061260, -4.5299463, 4.5299463
4: -2.1544065, 1.8428237, -2.1544065, 1.8428237, -3.9972303, 3.9972303

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079
time: 0.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.48 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169196
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9209028
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9156888
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9233022
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9149989, upper bound: 1.9099031
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.48
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209029, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204114
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9156888
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4772267, 1.2162780, -0.5006113, 1.2452075, -1.7224343, 1.7168893
1: -0.5837069, 1.9907751, -0.6059399, 2.0574260, -2.6411328, 2.5967150
2: -1.2414637, 1.3113892, -1.3041267, 1.3373374, -2.5788012, 2.6155159
3: -0.8910899, 2.9134102, -0.9207745, 3.0303936, -3.9214835, 3.8341846
4: -1.6892080, 1.4641249, -1.7729893, 1.4975882, -3.1867962, 3.2371142

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9130443, upper bound: 1.9251727
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5006113, 1.2452075, -1.8129137, 1.8252530
1: -0.6357365, 1.9145675, -0.6059399, 2.0574260, -2.6931624, 2.5205073
2: -1.3678999, 1.4330735, -1.3041267, 1.3373374, -2.7052374, 2.7372003
3: -0.9600172, 2.9499054, -0.9207745, 3.0303936, -3.9904108, 3.8706799
4: -1.7817144, 1.5861579, -1.7729893, 1.4975882, -3.2793026, 3.3591471

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251727
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290693, 1.4100626, -0.6191602, 1.3756136, -2.0046830, 2.0292227
1: -0.6945271, 2.0676351, -0.6878209, 2.0247331, -2.7192602, 2.7554560
2: -1.4875250, 1.5175171, -1.4858418, 1.4780548, -2.9655797, 3.0033588
3: -1.0283484, 3.1668901, -1.0179358, 3.1737194, -4.2020679, 4.1848259
4: -1.9433708, 1.6934371, -1.9349661, 1.6475326, -3.5909033, 3.6284032

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6293340, 1.4103444, -2.0393908, 2.0393257
1: -0.6945271, 2.0676394, -0.6947789, 2.0680566, -2.7625837, 2.7624183
2: -1.4875946, 1.5174351, -1.4880743, 1.5177747, -3.0053692, 3.0055094
3: -1.0283990, 3.1669998, -1.0287070, 3.1677570, -4.1961560, 4.1957068
4: -1.9434557, 1.6933250, -1.9440632, 1.6937697, -3.6372254, 3.6373882

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161977
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6293340, 1.4103444, -2.1322105, 2.1363587
1: -0.7863536, 2.1740503, -0.6947789, 2.0680566, -2.8544102, 2.8688293
2: -1.6623611, 1.6327487, -1.4880743, 1.5177747, -3.1801357, 3.1208229
3: -1.1238203, 3.4061260, -1.0287070, 3.1677570, -4.2915773, 4.4348330
4: -2.1544065, 1.8428237, -1.9440632, 1.6937697, -3.8481762, 3.7868869

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5070248, -2.1360712, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1740503, -2.8685775, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6327487, -3.1203432, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4061260, -4.4345250, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8428237, -3.7862794, 3.8477316

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.7218661, 1.5070248, -2.2288909, 2.2288909
1: -0.7863536, 2.1740503, -0.7863536, 2.1740503, -2.9604039, 2.9604039
2: -1.6623611, 1.6327487, -1.6623611, 1.6327487, -3.2951097, 3.2951097
3: -1.1238203, 3.4061260, -1.1238203, 3.4061260, -4.5299463, 4.5299463
4: -2.1544065, 1.8428237, -2.1544065, 1.8428237, -3.9972303, 3.9972303

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079
time: 0.35 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.55 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9209029, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204114
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9156888
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9130443, upper bound: 1.9251727
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251727
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161977
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169198
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169196
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169196
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9209026
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9156888
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4772267, 1.2162780, -0.5006113, 1.2452075, -1.7224343, 1.7168893
1: -0.5837069, 1.9907751, -0.6059399, 2.0574260, -2.6411328, 2.5967150
2: -1.2414637, 1.3113892, -1.3041267, 1.3373374, -2.5788012, 2.6155159
3: -0.8910899, 2.9134102, -0.9207745, 3.0303936, -3.9214835, 3.8341846
4: -1.6892080, 1.4641249, -1.7729893, 1.4975882, -3.1867962, 3.2371142

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 28

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121948, upper bound: 1.9063116
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9119353, upper bound: 1.9176267
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4772267, 1.2162780, -0.5502033, 1.2859328, -1.7631595, 1.7664813
1: -0.5837069, 1.9907751, -0.6213336, 1.9295921, -2.5132990, 2.6121087
2: -1.2414637, 1.3113892, -1.3427572, 1.4046347, -2.6460984, 2.6541464
3: -0.8910899, 2.9134102, -0.9433498, 2.9237518, -3.8148417, 3.8567600
4: -1.6892080, 1.4641249, -1.7625704, 1.5530781, -3.2422862, 3.2266953

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 28

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121948, upper bound: 1.9063114
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9119353, upper bound: 1.9176270
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5006113, 1.2452075, -1.8129137, 1.8252530
1: -0.6357365, 1.9145675, -0.6059399, 2.0574260, -2.6931624, 2.5205073
2: -1.3678999, 1.4330735, -1.3041267, 1.3373374, -2.7052374, 2.7372003
3: -0.9600172, 2.9499054, -0.9207745, 3.0303936, -3.9904108, 3.8706799
4: -1.7817144, 1.5861579, -1.7729893, 1.4975882, -3.2793026, 3.3591471

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176267
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5502033, 1.2859328, -1.8536389, 1.8748450
1: -0.6357365, 1.9145675, -0.6213336, 1.9295921, -2.5653286, 2.5359011
2: -1.3678999, 1.4330735, -1.3427572, 1.4046347, -2.7725346, 2.7758307
3: -0.9600172, 2.9499054, -0.9433498, 2.9237518, -3.8837690, 3.8932552
4: -1.7817144, 1.5861579, -1.7625704, 1.5530781, -3.3347926, 3.3487282

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5404360, 1.3013850, -0.4847693, 1.2275558, -1.7679918, 1.7861544
1: -0.6152678, 1.9276924, -0.5906477, 2.0254970, -2.6407647, 2.5183401
2: -1.3080988, 1.4132724, -1.2678347, 1.3213601, -2.6294589, 2.6811070
3: -0.9374819, 2.8937511, -0.9001975, 2.9715080, -3.9089899, 3.7939487
4: -1.7344885, 1.5620285, -1.7272196, 1.4779277, -3.2124162, 3.2892480

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9233023
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6181140, 1.3958161, -0.6191602, 1.3756136, -1.9937276, 2.0149763
1: -0.6849847, 2.0469999, -0.6878209, 2.0247331, -2.7097178, 2.7348208
2: -1.4684038, 1.5033507, -1.4858418, 1.4780548, -2.9464586, 2.9891925
3: -1.0164871, 3.1362658, -1.0179358, 3.1737194, -4.1902065, 4.1542015
4: -1.9182425, 1.6757677, -1.9349661, 1.6475326, -3.5657749, 3.6107337

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9149989, upper bound: 1.9099031
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6293340, 1.4103444, -2.0393908, 2.0393257
1: -0.6945271, 2.0676394, -0.6947789, 2.0680566, -2.7625837, 2.7624183
2: -1.4875946, 1.5174351, -1.4880743, 1.5177747, -3.0053692, 3.0055094
3: -1.0283990, 3.1669998, -1.0287070, 3.1677570, -4.1961560, 4.1957068
4: -1.9434557, 1.6933250, -1.9440632, 1.6937697, -3.6372254, 3.6373882

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9281774
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5069630, -2.1360095, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1736212, -2.8681483, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6324198, -3.1200144, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4058056, -4.4342046, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8426332, -3.7860889, 3.8477316

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9281774
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6293340, 1.4103444, -2.1322105, 2.1363587
1: -0.7863536, 2.1740503, -0.6947789, 2.0680566, -2.8544102, 2.8688293
2: -1.6623611, 1.6327487, -1.4880743, 1.5177747, -3.1801357, 3.1208229
3: -1.1238203, 3.4061260, -1.0287070, 3.1677570, -4.2915773, 4.4348330
4: -2.1544065, 1.8428237, -1.9440632, 1.6937697, -3.8481762, 3.7868869

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.7218661, 1.5069630, -2.2288291, 2.2288909
1: -0.7863536, 2.1740503, -0.7863536, 2.1736212, -2.9599748, 2.9604039
2: -1.6623611, 1.6327487, -1.6623611, 1.6324198, -3.2947810, 3.2951097
3: -1.1238203, 3.4061260, -1.1238203, 3.4058056, -4.5296259, 4.5299463
4: -2.1544065, 1.8428237, -2.1544065, 1.8426332, -3.9970398, 3.9972303

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6290464, 1.4099917, -2.0390382, 2.0390382
1: -0.6945271, 2.0676394, -0.6945271, 2.0676394, -2.7621665, 2.7621665
2: -1.4875946, 1.5174351, -1.4875946, 1.5174351, -3.0050297, 3.0050297
3: -1.0283990, 3.1669998, -1.0283990, 3.1669998, -4.1953988, 4.1953988
4: -1.9434557, 1.6933250, -1.9434557, 1.6933250, -3.6367807, 3.6367807

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9281774
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5070248, -2.1360712, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1740503, -2.8685775, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6327487, -3.1203432, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4061260, -4.4345250, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8428237, -3.7862794, 3.8477316

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9282064
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6290464, 1.4099917, -2.1318579, 2.1360712
1: -0.7863536, 2.1740503, -0.6945271, 2.0676394, -2.8539929, 2.8685775
2: -1.6623611, 1.6327487, -1.4875946, 1.5174351, -3.1797962, 3.1203432
3: -1.1238203, 3.4061260, -1.0283990, 3.1669998, -4.2908201, 4.4345250
4: -2.1544065, 1.8428237, -1.9434557, 1.6933250, -3.8477316, 3.7862794

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.7218661, 1.5070248, -2.2288909, 2.2288909
1: -0.7863536, 2.1740503, -0.7863536, 2.1740503, -2.9604039, 2.9604039
2: -1.6623611, 1.6327487, -1.6623611, 1.6327487, -3.2951097, 3.2951097
3: -1.1238203, 3.4061260, -1.1238203, 3.4061260, -4.5299463, 4.5299463
4: -2.1544065, 1.8428237, -2.1544065, 1.8428237, -3.9972303, 3.9972303

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183079
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079
time: 0.45 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.69 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169198
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169196
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9169196
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9209026
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9111746, upper bound: 1.9156888
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9121948, upper bound: 1.9063116
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9119353, upper bound: 1.9176267
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9121948, upper bound: 1.9063114
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9119353, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176267
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9233023
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9149989, upper bound: 1.9099031
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9281774
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9281774
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9281774
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9282064
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183079
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9183079

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209029, upper bound: 1.9187324
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9209029, upper bound: 1.9187324
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4841455, -2.0204530, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1454496, -2.7272899, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6076518, -2.8774037, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3772249, -4.2662373, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8141474, -3.5383635, 3.7739778

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4841455, -2.1129379, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1454496, -2.8296199, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6076518, -3.0875516, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3772249, -4.3697414, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8141474, -3.8048935, 3.9412985

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204114
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.5363073, 1.3928900, -2.0219364, 1.9462991
1: -0.6945271, 2.0676394, -0.5818403, 2.0295284, -2.7240555, 2.6494796
2: -1.4875946, 1.5174351, -1.2697520, 1.5284014, -3.0159960, 2.7871871
3: -1.0283990, 3.1669998, -0.8890121, 2.7896709, -3.8180699, 4.0560122
4: -1.9434557, 1.6933250, -1.7242160, 1.6678995, -3.6113553, 3.4175410

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6287925, 1.4827141, -2.1117606, 2.0387843
1: -0.6945271, 2.0676394, -0.6841701, 2.1662390, -2.8607662, 2.7518096
2: -1.4875946, 1.5174351, -1.4798999, 1.6526315, -3.1402261, 2.9973350
3: -1.0283990, 3.1669998, -0.9925163, 3.1035147, -4.1319137, 4.1595163
4: -1.9434557, 1.6933250, -1.9907460, 1.8352203, -3.7786760, 3.6840711

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6287925, 1.4827141, -2.2045803, 2.1357555
1: -0.7863536, 2.1736212, -0.6841701, 2.1662390, -2.9525926, 2.8577914
2: -1.6623611, 1.6324198, -1.4798999, 1.6526315, -3.3149927, 3.1123197
3: -1.1238203, 3.4058056, -0.9925163, 3.1035147, -4.2273350, 4.3983221
4: -2.1544065, 1.8426332, -1.9907460, 1.8352203, -3.9896269, 3.8333793

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.5363073, 1.3928900, -2.0222239, 1.9466517
1: -0.6947789, 2.0680566, -0.5818403, 2.0295284, -2.7243073, 2.6498969
2: -1.4880743, 1.5177747, -1.2697520, 1.5284014, -3.0164757, 2.7875266
3: -1.0287070, 3.1677570, -0.8890121, 2.7896709, -3.8183780, 4.0567694
4: -1.9440632, 1.6937697, -1.7242160, 1.6678995, -3.6119628, 3.4179857

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.5363073, 1.3928900, -2.1147561, 2.0432703
1: -0.7863536, 2.1736212, -0.5818403, 2.0295284, -2.8158820, 2.7554615
2: -1.6623611, 1.6324198, -1.2697520, 1.5284014, -3.1907625, 2.9021719
3: -1.1238203, 3.4058056, -0.8890121, 2.7896709, -3.9134912, 4.2948179
4: -2.1544065, 1.8426332, -1.7242160, 1.6678995, -3.8223062, 3.5668492

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6293340, 1.4103444, -0.6287925, 1.4827141, -2.1120481, 2.0391369
1: -0.6947789, 2.0680566, -0.6841701, 2.1662390, -2.8610179, 2.7522268
2: -1.4880743, 1.5177747, -1.4798999, 1.6526315, -3.1407058, 2.9976745
3: -1.0287070, 3.1677570, -0.9925163, 3.1035147, -4.1322217, 4.1602736
4: -1.9440632, 1.6937697, -1.9907460, 1.8352203, -3.7792835, 3.6845157

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9156888
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4426365, 1.1751368, -0.5006113, 1.2452075, -1.6878440, 1.6757481
1: -0.5514874, 1.8719301, -0.6059399, 2.0574260, -2.6089134, 2.4778700
2: -1.1483974, 1.2748471, -1.3041267, 1.3373374, -2.4857349, 2.5789738
3: -0.8542614, 2.7291231, -0.9207745, 3.0303936, -3.8846550, 3.6498976
4: -1.5596056, 1.4160752, -1.7729893, 1.4975882, -3.0571938, 3.1890645

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9122813, upper bound: 1.9243465
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9107777, upper bound: 1.9176270
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4426365, 1.1751368, -0.5502033, 1.2859328, -1.7285693, 1.7253401
1: -0.5514874, 1.8719301, -0.6213336, 1.9295921, -2.4810796, 2.4932637
2: -1.1483974, 1.2748471, -1.3427572, 1.4046347, -2.5530322, 2.6176043
3: -0.8542614, 2.7291231, -0.9433498, 2.9237518, -3.7780132, 3.6724730
4: -1.5596056, 1.4160752, -1.7625704, 1.5530781, -3.1126838, 3.1786456

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9107777, upper bound: 1.9176270
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9107777, upper bound: 1.9176270
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5006113, 1.2452075, -1.7119071, 1.6979122
1: -0.5722542, 1.9478822, -0.6059399, 2.0574260, -2.6296802, 2.5538220
2: -1.2166643, 1.2926129, -1.3041267, 1.3373374, -2.5540018, 2.5967398
3: -0.8774633, 2.8582191, -0.9207745, 3.0303936, -3.9078569, 3.7789936
4: -1.6534157, 1.4405326, -1.7729893, 1.4975882, -3.1510038, 3.2135220

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9130443, upper bound: 1.9251727
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5006113, 1.2452075, -1.8129137, 1.8252530
1: -0.6357365, 1.9145675, -0.6059399, 2.0574260, -2.6931624, 2.5205073
2: -1.3678999, 1.4330735, -1.3041267, 1.3373374, -2.7052374, 2.7372003
3: -0.9600172, 2.9499054, -0.9207745, 3.0303936, -3.9904108, 3.8706799
4: -1.7817144, 1.5861579, -1.7729893, 1.4975882, -3.2793026, 3.3591471

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251727
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5502033, 1.2859328, -1.7526324, 1.7475042
1: -0.5722542, 1.9478822, -0.6213336, 1.9295921, -2.5018463, 2.5692158
2: -1.2166643, 1.2926129, -1.3427572, 1.4046347, -2.6212990, 2.6353703
3: -0.8774633, 2.8582191, -0.9433498, 2.9237518, -3.8012152, 3.8015690
4: -1.6534157, 1.4405326, -1.7625704, 1.5530781, -3.2064939, 3.2031031

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176267
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5502033, 1.2859328, -1.8536389, 1.8748450
1: -0.6357365, 1.9145675, -0.6213336, 1.9295921, -2.5653286, 2.5359011
2: -1.3678999, 1.4330735, -1.3427572, 1.4046347, -2.7725346, 2.7758307
3: -0.9600172, 2.9499054, -0.9433498, 2.9237518, -3.8837690, 3.8932552
4: -1.7817144, 1.5861579, -1.7625704, 1.5530781, -3.3347926, 3.3487282

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4771729, 1.2161772, -0.4847693, 1.2275558, -1.7047286, 1.7009466
1: -0.5836568, 1.9906149, -0.5906477, 2.0254970, -2.6091537, 2.5812626
2: -1.2413397, 1.3112973, -1.2678347, 1.3213601, -2.5626998, 2.5791321
3: -0.8910279, 2.9131565, -0.9001975, 2.9715080, -3.8625360, 3.8133540
4: -1.6890316, 1.4640067, -1.7272196, 1.4779277, -3.1669593, 3.1912262

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9130443, upper bound: 1.9251726
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.4847693, 1.2275558, -1.7952619, 1.8094110
1: -0.6357365, 1.9145675, -0.5906477, 2.0254970, -2.6612334, 2.5052152
2: -1.3678999, 1.4330735, -1.2678347, 1.3213601, -2.6892600, 2.7009082
3: -0.9600172, 2.9499054, -0.9001975, 2.9715080, -3.9315252, 3.8501029
4: -1.7817144, 1.5861579, -1.7272196, 1.4779277, -3.2596421, 3.3133774

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251727
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290693, 1.4100626, -0.6191602, 1.3756136, -2.0046830, 2.0292227
1: -0.6945271, 2.0676351, -0.6878209, 2.0247331, -2.7192602, 2.7554560
2: -1.4875250, 1.5175171, -1.4858418, 1.4780548, -2.9655797, 3.0033588
3: -1.0283484, 3.1668901, -1.0179358, 3.1737194, -4.2020679, 4.1848259
4: -1.9433708, 1.6934371, -1.9349661, 1.6475326, -3.5909033, 3.6284032

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6293340, 1.4103444, -2.0393908, 2.0393257
1: -0.6945271, 2.0676394, -0.6947789, 2.0680566, -2.7625837, 2.7624183
2: -1.4875946, 1.5174351, -1.4880743, 1.5177747, -3.0053692, 3.0055094
3: -1.0283990, 3.1669998, -1.0287070, 3.1677570, -4.1961560, 4.1957068
4: -1.9434557, 1.6933250, -1.9440632, 1.6937697, -3.6372254, 3.6373882

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6293340, 1.4103444, -2.1322105, 2.1362970
1: -0.7863536, 2.1736212, -0.6947789, 2.0680566, -2.8544102, 2.8684001
2: -1.6623611, 1.6324198, -1.4880743, 1.5177747, -3.1801357, 3.1204941
3: -1.1238203, 3.4058056, -1.0287070, 3.1677570, -4.2915773, 4.4345126
4: -2.1544065, 1.8426332, -1.9440632, 1.6937697, -3.8481762, 3.7866964

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5069630, -2.1360095, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1736212, -2.8681483, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6324198, -3.1200144, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4058056, -4.4342046, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8426332, -3.7860889, 3.8477316

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.7218661, 1.5069630, -2.2288291, 2.2288291
1: -0.7863536, 2.1736212, -0.7863536, 2.1736212, -2.9599748, 2.9599748
2: -1.6623611, 1.6324198, -1.6623611, 1.6324198, -3.2947810, 3.2947810
3: -1.1238203, 3.4058056, -1.1238203, 3.4058056, -4.5296259, 4.5296259
4: -2.1544065, 1.8426332, -2.1544065, 1.8426332, -3.9970398, 3.9970398

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6293340, 1.4103444, -2.0393908, 2.0393257
1: -0.6945271, 2.0676394, -0.6947789, 2.0680566, -2.7625837, 2.7624183
2: -1.4875946, 1.5174351, -1.4880743, 1.5177747, -3.0053692, 3.0055094
3: -1.0283990, 3.1669998, -1.0287070, 3.1677570, -4.1961560, 4.1957068
4: -1.9434557, 1.6933250, -1.9440632, 1.6937697, -3.6372254, 3.6373882

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6293340, 1.4103444, -2.1322105, 2.1363587
1: -0.7863536, 2.1740503, -0.6947789, 2.0680566, -2.8544102, 2.8688293
2: -1.6623611, 1.6327487, -1.4880743, 1.5177747, -3.1801357, 3.1208229
3: -1.1238203, 3.4061260, -1.0287070, 3.1677570, -4.2915773, 4.4348330
4: -2.1544065, 1.8428237, -1.9440632, 1.6937697, -3.8481762, 3.7868869

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161977
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5069630, -2.1360095, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1736212, -2.8681483, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6324198, -3.1200144, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4058056, -4.4342046, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8426332, -3.7860889, 3.8477316

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.7218661, 1.5069630, -2.2288291, 2.2288909
1: -0.7863536, 2.1740503, -0.7863536, 2.1736212, -2.9599748, 2.9604039
2: -1.6623611, 1.6327487, -1.6623611, 1.6324198, -3.2947810, 3.2951097
3: -1.1238203, 3.4061260, -1.1238203, 3.4058056, -4.5296259, 4.5299463
4: -2.1544065, 1.8428237, -2.1544065, 1.8426332, -3.9970398, 3.9972303

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6290464, 1.4099917, -2.0390382, 2.0390382
1: -0.6945271, 2.0676394, -0.6945271, 2.0676394, -2.7621665, 2.7621665
2: -1.4875946, 1.5174351, -1.4875946, 1.5174351, -3.0050297, 3.0050297
3: -1.0283990, 3.1669998, -1.0283990, 3.1669998, -4.1953988, 4.1953988
4: -1.9434557, 1.6933250, -1.9434557, 1.6933250, -3.6367807, 3.6367807

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.6290464, 1.4099917, -2.1318579, 2.1360095
1: -0.7863536, 2.1736212, -0.6945271, 2.0676394, -2.8539929, 2.8681483
2: -1.6623611, 1.6324198, -1.4875946, 1.5174351, -3.1797962, 3.1200144
3: -1.1238203, 3.4058056, -1.0283990, 3.1669998, -4.2908201, 4.4342046
4: -2.1544065, 1.8426332, -1.9434557, 1.6933250, -3.8477316, 3.7860889

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5070248, -2.1360712, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1740503, -2.8685775, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6327487, -3.1203432, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4061260, -4.4345250, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8428237, -3.7862794, 3.8477316

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183078
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5069630, -0.7218661, 1.5070248, -2.2288909, 2.2288291
1: -0.7863536, 2.1736212, -0.7863536, 2.1740503, -2.9604039, 2.9599748
2: -1.6623611, 1.6324198, -1.6623611, 1.6327487, -3.2951097, 3.2947810
3: -1.1238203, 3.4058056, -1.1238203, 3.4061260, -4.5299463, 4.5296259
4: -2.1544065, 1.8426332, -2.1544065, 1.8428237, -3.9972303, 3.9970398

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.6290464, 1.4099917, -2.0390382, 2.0390382
1: -0.6945271, 2.0676394, -0.6945271, 2.0676394, -2.7621665, 2.7621665
2: -1.4875946, 1.5174351, -1.4875946, 1.5174351, -3.0050297, 3.0050297
3: -1.0283990, 3.1669998, -1.0283990, 3.1669998, -4.1953988, 4.1953988
4: -1.9434557, 1.6933250, -1.9434557, 1.6933250, -3.6367807, 3.6367807

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6290464, 1.4099917, -2.1318579, 2.1360712
1: -0.7863536, 2.1740503, -0.6945271, 2.0676394, -2.8539929, 2.8685775
2: -1.6623611, 1.6327487, -1.4875946, 1.5174351, -3.1797962, 3.1203432
3: -1.1238203, 3.4061260, -1.0283990, 3.1669998, -4.2908201, 4.4345250
4: -2.1544065, 1.8428237, -1.9434557, 1.6933250, -3.8477316, 3.7862794

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161977
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6290464, 1.4099917, -0.7218661, 1.5070248, -2.1360712, 2.1318579
1: -0.6945271, 2.0676394, -0.7863536, 2.1740503, -2.8685775, 2.8539929
2: -1.4875946, 1.5174351, -1.6623611, 1.6327487, -3.1203432, 3.1797962
3: -1.0283990, 3.1669998, -1.1238203, 3.4061260, -4.4345250, 4.2908201
4: -1.9434557, 1.6933250, -2.1544065, 1.8428237, -3.7862794, 3.8477316

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.7218661, 1.5070248, -2.2288909, 2.2288909
1: -0.7863536, 2.1740503, -0.7863536, 2.1740503, -2.9604039, 2.9604039
2: -1.6623611, 1.6327487, -1.6623611, 1.6327487, -3.2951097, 3.2951097
3: -1.1238203, 3.4061260, -1.1238203, 3.4061260, -4.5299463, 4.5299463
4: -2.1544065, 1.8428237, -2.1544065, 1.8428237, -3.9972303, 3.9972303

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9183079
time: 0.42 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.84 seconds
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9209029, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204115
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9169198, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9209029, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9204114
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9187324
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204114
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9201501, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9099164
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9187324, upper bound: 1.9156888
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9122813, upper bound: 1.9243465
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9107777, upper bound: 1.9176270
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9107777, upper bound: 1.9176270
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9107777, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9130443, upper bound: 1.9251727
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251727
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233023
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176267
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9130443, upper bound: 1.9251726
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9115793, upper bound: 1.9176270
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156885, upper bound: 1.9251727
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9233022
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9152623, upper bound: 1.9265876
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161977
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161977
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183078
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9275762, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161977
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9179445, upper bound: 1.9183080
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9161978
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.84
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9183079

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6228728, 1.4051275, -1.9414349, 2.0157628
1: -0.5818403, 2.0295284, -0.6888204, 2.0616698, -2.6435101, 2.7183487
2: -1.2697520, 1.5284014, -1.4792109, 1.5118654, -2.7816174, 3.0076122
3: -0.8890121, 2.7896709, -1.0259228, 3.1624193, -4.0514317, 3.8155937
4: -1.7242160, 1.6678995, -1.9344130, 1.6872003, -3.4114163, 3.6023126

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6228728, 1.4051275, -2.0339200, 2.1055870
1: -0.6841701, 2.1662390, -0.6888204, 2.0616698, -2.7458401, 2.8550594
2: -1.4798999, 1.6526315, -1.4792109, 1.5118654, -2.9917653, 3.1318424
3: -0.9925163, 3.1035147, -1.0259228, 3.1624193, -4.1549358, 4.1294374
4: -1.9907460, 1.8352203, -1.9344130, 1.6872003, -3.6779463, 3.7696333

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6229672, 1.4052701, -1.9415774, 2.0158572
1: -0.5818403, 2.0295284, -0.6889124, 2.0618849, -2.6437252, 2.7184408
2: -1.2697520, 1.5284014, -1.4794292, 1.5119994, -2.7817514, 3.0078306
3: -0.8890121, 2.7896709, -1.0260291, 3.1628151, -4.0518274, 3.8157001
4: -1.7242160, 1.6678995, -1.9346943, 1.6873670, -3.4115829, 3.6025939

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6941342, 1.4840844, -2.0203917, 2.0870242
1: -0.5818403, 2.0295284, -0.7596393, 2.1450229, -2.7268631, 2.7891676
2: -1.2697520, 1.5284014, -1.6208410, 1.6073264, -2.8770785, 3.1492424
3: -0.8890121, 2.7896709, -1.1085167, 3.3769073, -4.2659197, 3.8981876
4: -1.7242160, 1.6678995, -2.1060781, 1.8139427, -3.5381587, 3.7739778

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9201501
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6229672, 1.4052701, -2.0340626, 2.1056814
1: -0.6841701, 2.1662390, -0.6889124, 2.0618849, -2.7460551, 2.8551514
2: -1.4798999, 1.6526315, -1.4794292, 1.5119994, -2.9918993, 3.1320608
3: -0.9925163, 3.1035147, -1.0260291, 3.1628151, -4.1553316, 4.1295438
4: -1.9907460, 1.8352203, -1.9346943, 1.6873670, -3.6781130, 3.7699146

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9187324
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6941342, 1.4840844, -2.1128769, 2.1768484
1: -0.6841701, 2.1662390, -0.7596393, 2.1450229, -2.8291931, 2.9258783
2: -1.4798999, 1.6526315, -1.6208410, 1.6073264, -3.0872264, 3.2734725
3: -0.9925163, 3.1035147, -1.1085167, 3.3769073, -4.3694239, 4.2120314
4: -1.9907460, 1.8352203, -2.1060781, 1.8139427, -3.8046887, 3.9412985

Time for backsubstitution: 1.64 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=2.269134044647217
rel_dist={0: [-1.9301230654393355, 1.9301230654393358]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178776, upper bound: 1.9201370
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269452, upper bound: 1.9269451
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -1.9178776, upper bound: 1.9201370
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -1.9269452, upper bound: 1.9269451

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.5508800, 1.3501766, -1.9304756, 1.9951298
1: -0.6313032, 2.1405544, -0.6288738, 2.0743403, -2.7056437, 2.7694283
2: -1.3848062, 1.5734707, -1.3720207, 1.4563724, -2.8411784, 2.9454913
3: -0.9428792, 3.0088511, -0.9553370, 3.0639458, -4.0068250, 3.9641881
4: -1.8759956, 1.7349151, -1.8440752, 1.6113482, -3.4873438, 3.5789905

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9168834, upper bound: 1.9168834
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9168834, upper bound: 1.9201370
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7438335, 1.5253004, -2.2143774, 2.2126141
1: -0.7561793, 2.1721611, -0.8104258, 2.3001223, -3.0563016, 2.9825869
2: -1.6216621, 1.5724510, -1.7379694, 1.6217314, -3.2433934, 3.3104205
3: -1.0972075, 3.4126759, -1.1569667, 3.6523972, -4.7496047, 4.5696425
4: -2.1124964, 1.7667136, -2.2834320, 1.8319958, -3.9444923, 4.0501456

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201370, upper bound: 1.9178776
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9201370, upper bound: 1.9269452
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.08 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.9168834, upper bound: 1.9168834
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.9168834, upper bound: 1.9201370
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.9201370, upper bound: 1.9178776
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -1.9201370, upper bound: 1.9269452

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.6449399, 1.4331532, -2.0134521, 2.0891898
1: -0.6313032, 2.1405544, -0.7115011, 2.1302786, -2.7615819, 2.8520555
2: -1.3848062, 1.5734707, -1.5320005, 1.5386055, -2.9234116, 3.1054711
3: -0.9428792, 3.0088511, -1.0510054, 3.2833786, -4.2262578, 4.0598564
4: -1.8759956, 1.7349151, -2.0118685, 1.7234324, -3.5994282, 3.7467837

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190335
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9147088, upper bound: 1.9151113
time: 0.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.5802990, 1.4442499, -2.1333268, 2.0490797
1: -0.7561793, 2.1721611, -0.6313032, 2.1405544, -2.8967338, 2.8034644
2: -1.6216621, 1.5724510, -1.3848062, 1.5734707, -3.1951327, 2.9572573
3: -1.0972075, 3.4126759, -0.9428792, 3.0088511, -4.1060586, 4.3555551
4: -2.1124964, 1.7667136, -1.8759956, 1.7349151, -3.8474116, 3.6427093

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9168833, upper bound: 1.9163591
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187696, upper bound: 1.9148831
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.6890769, 1.4687806, -2.1578574, 2.1578574
1: -0.7561793, 2.1721611, -0.7561793, 2.1721611, -2.9283404, 2.9283404
2: -1.6216621, 1.5724510, -1.6216621, 1.5724510, -3.1941133, 3.1941133
3: -1.0972075, 3.4126759, -1.0972075, 3.4126759, -4.5098834, 4.5098834
4: -2.1124964, 1.7667136, -2.1124964, 1.7667136, -3.8792100, 3.8792100

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9168833, upper bound: 1.9163591
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187696, upper bound: 1.9177305
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.65 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190335
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -1.9147088, upper bound: 1.9151113
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -1.9168833, upper bound: 1.9163591
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -1.9187696, upper bound: 1.9148831
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -1.9168833, upper bound: 1.9163591
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -1.9187696, upper bound: 1.9177305

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6449399, 1.4331532, -1.9694605, 2.0378299
1: -0.5818403, 2.0295284, -0.7115011, 2.1302786, -2.7121189, 2.7410295
2: -1.2697520, 1.5284014, -1.5320005, 1.5386055, -2.8083575, 3.0604019
3: -0.8890121, 2.7896709, -1.0510054, 3.2833786, -4.1723909, 3.8406763
4: -1.7242160, 1.6678995, -2.0118685, 1.7234324, -3.4476485, 3.6797681

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5760635, 1.4389447, -2.1608109, 2.0830884
1: -0.7863536, 2.1740503, -0.6273289, 2.1307991, -2.9171526, 2.8013792
2: -1.6623611, 1.6327487, -1.3767133, 1.5677444, -3.2301054, 3.0094619
3: -1.1238203, 3.4061260, -0.9384818, 2.9952812, -4.1191015, 4.3446078
4: -2.1544065, 1.8428237, -1.8651457, 1.7277613, -3.8821678, 3.7079694

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6610966, 1.4372065, -2.1590726, 2.1681213
1: -0.7863536, 2.1740503, -0.7274408, 2.1185155, -2.9048691, 2.9014912
2: -1.6623611, 1.6327487, -1.5606995, 1.5407701, -3.2031312, 3.1934481
3: -1.1238203, 3.4061260, -1.0650692, 3.3075619, -4.4313822, 4.4711952
4: -2.1544065, 1.8428237, -2.0347166, 1.7254040, -3.8798106, 3.8775403

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9147438, upper bound: 1.9113780
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9147438, upper bound: 1.9094061
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.35 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.9147438, upper bound: 1.9113780
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.9147438, upper bound: 1.9094061

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190335
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.37 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190335
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.40 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.38 seconds
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.38
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.42 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.49 seconds
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.49
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.46 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.57 seconds
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.57
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.42 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 2.63 seconds
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.47 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.00 seconds
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9170209, upper bound: 1.9148831
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.00
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.50 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.5363073, 1.3928900, -2.1147561, 2.0433321
1: -0.7863536, 2.1740503, -0.5818403, 2.0295284, -2.8158820, 2.7558906
2: -1.6623611, 1.6327487, -1.2697520, 1.5284014, -3.1907625, 2.9025006
3: -1.1238203, 3.4061260, -0.8890121, 2.7896709, -3.9134912, 4.2951384
4: -2.1544065, 1.8428237, -1.7242160, 1.6678995, -3.8223062, 3.5670397

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.7218661, 1.5070248, -0.6287925, 1.4827141, -2.2045803, 2.1358173
1: -0.7863536, 2.1740503, -0.6841701, 2.1662390, -2.9525926, 2.8582206
2: -1.6623611, 1.6327487, -1.4798999, 1.6526315, -3.3149927, 3.1126485
3: -1.1238203, 3.4061260, -0.9925163, 3.1035147, -4.2273350, 4.3986425
4: -2.1544065, 1.8428237, -1.9907460, 1.8352203, -3.9896269, 3.8335698

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831
time: 0.41 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 2.85 seconds
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9170209
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9190334, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9099164
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 2.85
Output dim: 0, lower bound: -1.9189183, upper bound: 1.9148831

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6912913, 1.4776999, -2.0140071, 2.0841813
1: -0.5818403, 2.0295284, -0.7556162, 2.1379690, -2.7198093, 2.7851446
2: -1.2697520, 1.5284014, -1.6117010, 1.5996788, -2.8694308, 3.1401024
3: -0.8890121, 2.7896709, -1.1007905, 3.3630695, -4.2520819, 3.8904614
4: -1.7242160, 1.6678995, -2.0965319, 1.8041947, -3.5284107, 3.7644315

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190334
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9189183
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6912913, 1.4776999, -2.1064925, 2.1740055
1: -0.6841701, 2.1662390, -0.7556162, 2.1379690, -2.8221393, 2.9218552
2: -1.4798999, 1.6526315, -1.6117010, 1.5996788, -3.0795786, 3.2643325
3: -0.9925163, 3.1035147, -1.1007905, 3.3630695, -4.3555861, 4.2043052
4: -1.9907460, 1.8352203, -2.0965319, 1.8041947, -3.7949407, 3.9317522

Time for backsubstitution: 1.63 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=2.269134044647217
rel_dist={0: [-1.927168397752424, 1.9271683977524248]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1153.74 seconds
