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
execution time: IAR + LP analysis = 1.33 + 1.11 = 2.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.9315851, upper bound: 1.9315851


# Binary Search by BASE starts (time budget: 1197.56 seconds, max iter: 100)

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
Binary search time: 43.90 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1153.65 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -1.9262004, upper bound: 1.9232330
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -1.9314710, upper bound: 1.9314708

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.5802990, 1.4442499, -0.6770015, 1.4656314, -2.0459304, 2.1212513
1: -0.6313032, 2.1405544, -0.7475953, 2.2259817, -2.8572850, 2.8881497
2: -1.3848062, 1.5734707, -1.6076193, 1.5682487, -2.9530549, 3.1810899
3: -0.9428792, 3.0088511, -1.0890913, 3.4468250, -4.3897042, 4.0979424
4: -1.8759956, 1.7349151, -2.1315250, 1.7607409, -3.6367364, 3.8664403

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
time: 0.32 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
time: 0.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7438335, 1.5253004, -2.2143774, 2.2126141
1: -0.7561793, 2.1721611, -0.8104258, 2.3001223, -3.0563016, 2.9825869
2: -1.6216621, 1.5724510, -1.7379694, 1.6217314, -3.2433934, 3.3104205
3: -1.0972075, 3.4126759, -1.1569667, 3.6523972, -4.7496047, 4.5696425
4: -2.1124964, 1.7667136, -2.2834320, 1.8319958, -3.9444923, 4.0501456

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9304278, upper bound: 1.9183853
time: 0.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9195167
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9220986
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9157402, upper bound: 1.9207749
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9304278, upper bound: 1.9183853
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9195167

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.6770015, 1.4656314, -2.0019388, 2.0698915
1: -0.5818403, 2.0295284, -0.7475953, 2.2259817, -2.8078220, 2.7771237
2: -1.2697520, 1.5284014, -1.6076193, 1.5682487, -2.8380008, 3.1360207
3: -0.8890121, 2.7896709, -1.0890913, 3.4468250, -4.3358374, 3.8787622
4: -1.7242160, 1.6678995, -2.1315250, 1.7607409, -3.4849567, 3.7994246

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082928, upper bound: 1.9183955
time: 0.31 seconds

## Relational analysis of IS_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9209100
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065310, upper bound: 1.9146858
time: 0.29 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.6770015, 1.4656314, -2.0944238, 2.1597157
1: -0.6841701, 2.1662390, -0.7475953, 2.2259817, -2.9101520, 2.9138343
2: -1.4798999, 1.6526315, -1.6076193, 1.5682487, -3.0481486, 3.2602508
3: -0.9925163, 3.1035147, -1.0890913, 3.4468250, -4.4393415, 4.1926060
4: -1.9907460, 1.8352203, -2.1315250, 1.7607409, -3.7514868, 3.9667454

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9078184, upper bound: 1.9052402
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9134665, upper bound: 1.9164596
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9143525, upper bound: 1.9145881
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9133475, upper bound: 1.9148207
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.6723809, 1.4565330, -2.1456099, 2.1411614
1: -0.7561793, 2.1721611, -0.7381821, 2.1788821, -2.9350615, 2.9103432
2: -1.6216621, 1.5724510, -1.5778151, 1.5577803, -3.1794424, 3.1502662
3: -1.0972075, 3.4126759, -1.0751319, 3.3567352, -4.4539428, 4.4878078
4: -2.1124964, 1.7667136, -2.0781775, 1.7465845, -3.8590808, 3.8448911

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9277943, upper bound: 1.9152623
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7575412, 1.5481710, -2.2372479, 2.2263217
1: -0.7561793, 2.1721611, -0.8227930, 2.2892652, -3.0454445, 2.9949541
2: -1.6216621, 1.5724510, -1.7448044, 1.6674051, -3.2890673, 3.3172555
3: -1.0972075, 3.4126759, -1.1639147, 3.6009045, -4.6981120, 4.5765905
4: -2.1124964, 1.7667136, -2.2804823, 1.8911014, -4.0035977, 4.0471959

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164305
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9150023
time: 0.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.12 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9209100
IS_A1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9065310, upper bound: 1.9146858
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9143525, upper bound: 1.9145881
IS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9133475, upper bound: 1.9148207
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9277943, upper bound: 1.9152623
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164305
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9150023

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6770015, 1.4656314, -1.8754034, 1.9457053
1: -0.4681202, 1.8429868, -0.7475953, 2.2259817, -2.6941018, 2.5905821
2: -1.0267184, 1.4059389, -1.6076193, 1.5682487, -2.5949671, 3.0135581
3: -0.7750952, 2.4403634, -1.0890913, 3.4468250, -4.2219200, 3.5294547
4: -1.4172378, 1.5203712, -2.1315250, 1.7607409, -3.1779785, 3.6518962

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9209100
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2

### Relational analysis result of IS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6191602, 1.3756136, -0.6723809, 1.4565330, -2.0756931, 2.0479946
1: -0.6878209, 2.0247331, -0.7381821, 2.1788821, -2.8667030, 2.7629151
2: -1.4858418, 1.4780548, -1.5778151, 1.5577803, -3.0436220, 3.0558698
3: -1.0179358, 3.1737194, -1.0751319, 3.3567352, -4.3746710, 4.2488513
4: -1.9349661, 1.6475326, -2.0781775, 1.7465845, -3.6815505, 3.7257099

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9277943, upper bound: 1.9152623
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9277943, upper bound: 1.9152623
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.18 seconds
IS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9209100
IS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9277943, upper bound: 1.9152623
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9277943, upper bound: 1.9152623

## BFS IS instance: IS_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6770015, 1.4656314, -1.8754034, 1.9457053
1: -0.4681202, 1.8429868, -0.7475953, 2.2259817, -2.6941018, 2.5905821
2: -1.0267184, 1.4059389, -1.6076193, 1.5682487, -2.5949671, 3.0135581
3: -0.7750952, 2.4403634, -1.0890913, 3.4468250, -4.2219200, 3.5294547
4: -1.4172378, 1.5203712, -2.1315250, 1.7607409, -3.1779785, 3.6518962

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9166772
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082199, upper bound: 1.9206774
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082199, upper bound: 1.9209100
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6770015, 1.4656314, -1.9356928, 2.0053835
1: -0.5353262, 1.9529831, -0.7475953, 2.2259817, -2.7613080, 2.7005785
2: -1.1786942, 1.4957325, -1.6076193, 1.5682487, -2.7469430, 3.1033518
3: -0.8506749, 2.7089930, -1.0890913, 3.4468250, -4.2974997, 3.7980843
4: -1.6099305, 1.6463493, -2.1315250, 1.7607409, -3.3706713, 3.7778745

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.32 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.6723809, 1.4565330, -2.0242391, 1.9970226
1: -0.6357365, 1.9145675, -0.7381821, 2.1788821, -2.8146186, 2.6527495
2: -1.3678999, 1.4330735, -1.5778151, 1.5577803, -2.9256802, 3.0108886
3: -0.9600172, 2.9499054, -1.0751319, 3.3567352, -4.3167524, 4.0250373
4: -1.7817144, 1.5861579, -2.0781775, 1.7465845, -3.5282989, 3.6643353

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9150023
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9150023
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.6673212, 1.4338410, -0.6723809, 1.4565330, -2.1238542, 2.1062219
1: -0.7327099, 2.0694237, -0.7381821, 2.1788821, -2.9115920, 2.8076057
2: -1.5579519, 1.5491378, -1.5778151, 1.5577803, -3.1157322, 3.1269529
3: -1.0585232, 3.2334042, -1.0751319, 3.3567352, -4.4152584, 4.3085361
4: -2.0197039, 1.7408283, -2.0781775, 1.7465845, -3.7662883, 3.8190057

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9115881
time: 0.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.23 seconds
IS_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9082199, upper bound: 1.9206774
IS_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9082199, upper bound: 1.9209100
IS_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9150023
IS_A2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9150023
IS_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9150023, upper bound: 1.9115793
IS_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9115881

## BFS IS instance: IS_A1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.5104001, 1.2907361, -1.7005081, 1.7791039
1: -0.4681202, 1.8429868, -0.5915279, 2.0047708, -2.4728909, 2.4345148
2: -1.0267184, 1.4059389, -1.2737646, 1.3961782, -2.4228966, 2.6797035
3: -0.7750952, 2.4403634, -0.9141693, 2.9073734, -3.6824687, 3.3545327
4: -1.4172378, 1.5203712, -1.7245560, 1.5372548, -2.9544926, 3.2449272

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144532
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.5694652, 1.3358598, -1.7456317, 1.8381690
1: -0.4681202, 1.8429868, -0.6504865, 2.0509157, -2.5190358, 2.4934733
2: -1.0267184, 1.4059389, -1.4051085, 1.4416969, -2.4684153, 2.8110473
3: -0.7750952, 2.4403634, -0.9786949, 3.1067924, -3.8818877, 3.4190583
4: -1.4172378, 1.5203712, -1.8767214, 1.6027833, -3.0200210, 3.3970926

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146858
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.30 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.32 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7105598, 1.5058548, -1.9759163, 2.0389419
1: -0.5353262, 1.9529831, -0.7756906, 2.2401943, -2.7755206, 2.7286737
2: -1.1786942, 1.4957325, -1.6519613, 1.6299368, -2.8086309, 3.1476939
3: -0.8506749, 2.7089930, -1.1140494, 3.4695072, -4.3201818, 3.8230424
4: -1.6099305, 1.6463493, -2.1803675, 1.8414104, -3.4513409, 3.8267169

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6723809, 1.4565330, -2.0067363, 1.9583137
1: -0.6213336, 1.9295921, -0.7381821, 2.1788821, -2.8002157, 2.6677742
2: -1.3427572, 1.4046347, -1.5778151, 1.5577803, -2.9005375, 2.9824498
3: -0.9433498, 2.9237518, -1.0751319, 3.3567352, -4.3000851, 3.9988837
4: -1.7625704, 1.5530781, -2.0781775, 1.7465845, -3.5091548, 3.6312556

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.12 seconds
IS_A1_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144532
IS_A1_A1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
IS_A1_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146858
IS_A1_A1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
IS_A1_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622

## BFS IS instance: IS_A1_A1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6215014, 1.4085987, -1.8183706, 1.8902053
1: -0.4681202, 1.8429868, -0.6910777, 2.1192222, -2.5873423, 2.5340645
2: -1.0267184, 1.4059389, -1.4793415, 1.5146111, -2.5413294, 2.8852804
3: -0.7750952, 2.4403634, -1.0249138, 3.2062330, -3.9813282, 3.4652772
4: -1.4172378, 1.5203712, -1.9659233, 1.6891631, -3.1064010, 3.4862945

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.32 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.31 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7105598, 1.5058548, -1.9156268, 1.9792637
1: -0.4681202, 1.8429868, -0.7756906, 2.2401943, -2.7083144, 2.6186774
2: -1.0267184, 1.4059389, -1.6519613, 1.6299368, -2.6566553, 3.0579002
3: -0.7750952, 2.4403634, -1.1140494, 3.4695072, -4.2446022, 3.5544128
4: -1.4172378, 1.5203712, -2.1803675, 1.8414104, -3.2586482, 3.7007387

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
time: 0.31 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7105598, 1.5058548, -1.9759163, 2.0389419
1: -0.5353262, 1.9529831, -0.7756906, 2.2401943, -2.7755206, 2.7286737
2: -1.1786942, 1.4957325, -1.6519613, 1.6299368, -2.8086309, 3.1476939
3: -0.8506749, 2.7089930, -1.1140494, 3.4695072, -4.3201818, 3.8230424
4: -1.6099305, 1.6463493, -2.1803675, 1.8414104, -3.4513409, 3.8267169

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.31 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5167794, 1.2480004, -1.7982037, 1.8027122
1: -0.6213336, 1.9295921, -0.6187482, 2.0452437, -2.6665773, 2.5483403
2: -1.3427572, 1.4046347, -1.3220568, 1.3385221, -2.6812792, 2.7266915
3: -0.9433498, 2.9237518, -0.9353175, 3.0362320, -3.9795818, 3.8590693
4: -1.7625704, 1.5530781, -1.7887945, 1.4996266, -3.2621970, 3.3418727

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.16 seconds
IS_A1_A1_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
IS_A1_A1_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
IS_A1_A1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.16
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.32 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.35 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.31 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.31 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7105598, 1.5058548, -1.9759163, 2.0389419
1: -0.5353262, 1.9529831, -0.7756906, 2.2401943, -2.7755206, 2.7286737
2: -1.1786942, 1.4957325, -1.6519613, 1.6299368, -2.8086309, 3.1476939
3: -0.8506749, 2.7089930, -1.1140494, 3.4695072, -4.3201818, 3.8230424
4: -1.6099305, 1.6463493, -2.1803675, 1.8414104, -3.4513409, 3.8267169

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6723809, 1.4565330, -2.0067363, 1.9583137
1: -0.6213336, 1.9295921, -0.7381821, 2.1788821, -2.8002157, 2.6677742
2: -1.3427572, 1.4046347, -1.5778151, 1.5577803, -2.9005375, 2.9824498
3: -0.9433498, 2.9237518, -1.0751319, 3.3567352, -4.3000851, 3.9988837
4: -1.7625704, 1.5530781, -2.0781775, 1.7465845, -3.5091548, 3.6312556

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152623
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152623
time: 0.36 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.23 seconds
IS_A1_A1_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A1_A1_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.23
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152623

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6215014, 1.4085987, -1.8183706, 1.8902053
1: -0.4681202, 1.8429868, -0.6910777, 2.1192222, -2.5873423, 2.5340645
2: -1.0267184, 1.4059389, -1.4793415, 1.5146111, -2.5413294, 2.8852804
3: -0.7750952, 2.4403634, -1.0249138, 3.2062330, -3.9813282, 3.4652772
4: -1.4172378, 1.5203712, -1.9659233, 1.6891631, -3.1064010, 3.4862945

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.41 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7075887, 1.5035110, -1.9132829, 1.9762925
1: -0.4681202, 1.8429868, -0.7728319, 2.2361383, -2.7042584, 2.6158187
2: -1.0267184, 1.4059389, -1.6448193, 1.6279107, -2.6546292, 3.0507581
3: -0.7750952, 2.4403634, -1.1082888, 3.4591293, -4.2342243, 3.5486522
4: -1.4172378, 1.5203712, -2.1707525, 1.8386446, -3.2558823, 3.6911237

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6205978, 1.4072542, -1.8170261, 1.8893017
1: -0.4681202, 1.8429868, -0.6902347, 2.1172371, -2.5853572, 2.5332215
2: -1.0267184, 1.4059389, -1.4774981, 1.5133135, -2.5400319, 2.8834369
3: -0.7750952, 2.4403634, -1.0240054, 3.2029772, -3.9780724, 3.4643688
4: -1.4172378, 1.5203712, -1.9635334, 1.6875764, -3.1048141, 3.4839046

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.41 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7105598, 1.5058548, -1.9156268, 1.9792637
1: -0.4681202, 1.8429868, -0.7756906, 2.2401943, -2.7083144, 2.6186774
2: -1.0267184, 1.4059389, -1.6519613, 1.6299368, -2.6566553, 3.0579002
3: -0.7750952, 2.4403634, -1.1140494, 3.4695072, -4.2446022, 3.5544128
4: -1.4172378, 1.5203712, -2.1803675, 1.8414104, -3.2586482, 3.7007387

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7105598, 1.5058548, -1.9759163, 2.0389419
1: -0.5353262, 1.9529831, -0.7756906, 2.2401943, -2.7755206, 2.7286737
2: -1.1786942, 1.4957325, -1.6519613, 1.6299368, -2.8086309, 3.1476939
3: -0.8506749, 2.7089930, -1.1140494, 3.4695072, -4.3201818, 3.8230424
4: -1.6099305, 1.6463493, -2.1803675, 1.8414104, -3.4513409, 3.8267169

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.32 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5167794, 1.2480004, -1.7982037, 1.8027122
1: -0.6213336, 1.9295921, -0.6187482, 2.0452437, -2.6665773, 2.5483403
2: -1.3427572, 1.4046347, -1.3220568, 1.3385221, -2.6812792, 2.7266915
3: -0.9433498, 2.9237518, -0.9353175, 3.0362320, -3.9795818, 3.8590693
4: -1.7625704, 1.5530781, -1.7887945, 1.4996266, -3.2621970, 3.3418727

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.37 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.26 seconds
IS_A1_A1_A1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
IS_A1_A1_A1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
IS_A1_A1_A1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.26
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.36 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7105598, 1.5058548, -1.9759163, 2.0389419
1: -0.5353262, 1.9529831, -0.7756906, 2.2401943, -2.7755206, 2.7286737
2: -1.1786942, 1.4957325, -1.6519613, 1.6299368, -2.8086309, 3.1476939
3: -0.8506749, 2.7089930, -1.1140494, 3.4695072, -4.3201818, 3.8230424
4: -1.6099305, 1.6463493, -2.1803675, 1.8414104, -3.4513409, 3.8267169

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6723809, 1.4565330, -2.0067363, 1.9583137
1: -0.6213336, 1.9295921, -0.7381821, 2.1788821, -2.8002157, 2.6677742
2: -1.3427572, 1.4046347, -1.5778151, 1.5577803, -2.9005375, 2.9824498
3: -0.9433498, 2.9237518, -1.0751319, 3.3567352, -4.3000851, 3.9988837
4: -1.7625704, 1.5530781, -2.0781775, 1.7465845, -3.5091548, 3.6312556

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9127985
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5005965, 1.2284206, -1.7567037, 1.7813599
1: -0.6040082, 1.9247079, -0.6031599, 2.0058041, -2.6098123, 2.5278678
2: -1.3036880, 1.4001466, -1.2835169, 1.3217146, -2.6254027, 2.6836634
3: -0.9276228, 2.8863373, -0.9144611, 2.9694157, -3.8970385, 3.8007984
4: -1.7242384, 1.5467284, -1.7390747, 1.4786390, -3.2028775, 3.2858031

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9127985
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.6025977, 1.3614380, -1.8897212, 1.8833611
1: -0.6040082, 1.9247079, -0.6731129, 2.0198145, -2.6238227, 2.5978208
2: -1.3036880, 1.4001466, -1.4475107, 1.4631717, -2.7668598, 2.8476572
3: -0.9276228, 2.8863373, -0.9972734, 3.1262207, -4.0538435, 3.8836107
4: -1.7242384, 1.5467284, -1.9038792, 1.6280352, -3.3522735, 3.4506075

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
time: 0.38 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 2.35 seconds
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9153594, upper bound: 1.9164293
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9127985
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9127985
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.35
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9152622

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6215014, 1.4085987, -1.8183706, 1.8902053
1: -0.4681202, 1.8429868, -0.6910777, 2.1192222, -2.5873423, 2.5340645
2: -1.0267184, 1.4059389, -1.4793415, 1.5146111, -2.5413294, 2.8852804
3: -0.7750952, 2.4403634, -1.0249138, 3.2062330, -3.9813282, 3.4652772
4: -1.4172378, 1.5203712, -1.9659233, 1.6891631, -3.1064010, 3.4862945

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7075887, 1.5035110, -1.9132829, 1.9762925
1: -0.4681202, 1.8429868, -0.7728319, 2.2361383, -2.7042584, 2.6158187
2: -1.0267184, 1.4059389, -1.6448193, 1.6279107, -2.6546292, 3.0507581
3: -0.7750952, 2.4403634, -1.1082888, 3.4591293, -4.2342243, 3.5486522
4: -1.4172378, 1.5203712, -2.1707525, 1.8386446, -3.2558823, 3.6911237

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6205978, 1.4072542, -1.8170261, 1.8893017
1: -0.4681202, 1.8429868, -0.6902347, 2.1172371, -2.5853572, 2.5332215
2: -1.0267184, 1.4059389, -1.4774981, 1.5133135, -2.5400319, 2.8834369
3: -0.7750952, 2.4403634, -1.0240054, 3.2029772, -3.9780724, 3.4643688
4: -1.4172378, 1.5203712, -1.9635334, 1.6875764, -3.1048141, 3.4839046

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7075887, 1.5035110, -1.9132829, 1.9762925
1: -0.4681202, 1.8429868, -0.7728319, 2.2361383, -2.7042584, 2.6158187
2: -1.0267184, 1.4059389, -1.6448193, 1.6279107, -2.6546292, 3.0507581
3: -0.7750952, 2.4403634, -1.1082888, 3.4591293, -4.2342243, 3.5486522
4: -1.4172378, 1.5203712, -2.1707525, 1.8386446, -3.2558823, 3.6911237

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.40 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6205978, 1.4072542, -1.8170261, 1.8893017
1: -0.4681202, 1.8429868, -0.6902347, 2.1172371, -2.5853572, 2.5332215
2: -1.0267184, 1.4059389, -1.4774981, 1.5133135, -2.5400319, 2.8834369
3: -0.7750952, 2.4403634, -1.0240054, 3.2029772, -3.9780724, 3.4643688
4: -1.4172378, 1.5203712, -1.9635334, 1.6875764, -3.1048141, 3.4839046

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.39 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7075887, 1.5035110, -1.9132829, 1.9762925
1: -0.4681202, 1.8429868, -0.7728319, 2.2361383, -2.7042584, 2.6158187
2: -1.0267184, 1.4059389, -1.6448193, 1.6279107, -2.6546292, 3.0507581
3: -0.7750952, 2.4403634, -1.1082888, 3.4591293, -4.2342243, 3.5486522
4: -1.4172378, 1.5203712, -2.1707525, 1.8386446, -3.2558823, 3.6911237

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.40 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.6205978, 1.4072542, -1.8170261, 1.8893017
1: -0.4681202, 1.8429868, -0.6902347, 2.1172371, -2.5853572, 2.5332215
2: -1.0267184, 1.4059389, -1.4774981, 1.5133135, -2.5400319, 2.8834369
3: -0.7750952, 2.4403634, -1.0240054, 3.2029772, -3.9780724, 3.4643688
4: -1.4172378, 1.5203712, -1.9635334, 1.6875764, -3.1048141, 3.4839046

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9146458, upper bound: 1.9143787
time: 0.39 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.7105598, 1.5058548, -1.9156268, 1.9792637
1: -0.4681202, 1.8429868, -0.7756906, 2.2401943, -2.7083144, 2.6186774
2: -1.0267184, 1.4059389, -1.6519613, 1.6299368, -2.6566553, 3.0579002
3: -0.7750952, 2.4403634, -1.1140494, 3.4695072, -4.2446022, 3.5544128
4: -1.4172378, 1.5203712, -2.1803675, 1.8414104, -3.2586482, 3.7007387

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.39 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
time: 0.35 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7105598, 1.5058548, -1.9759163, 2.0389419
1: -0.5353262, 1.9529831, -0.7756906, 2.2401943, -2.7755206, 2.7286737
2: -1.1786942, 1.4957325, -1.6519613, 1.6299368, -2.8086309, 3.1476939
3: -0.8506749, 2.7089930, -1.1140494, 3.4695072, -4.3201818, 3.8230424
4: -1.6099305, 1.6463493, -2.1803675, 1.8414104, -3.4513409, 3.8267169

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.41 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6723809, 1.4565330, -1.9571443, 1.9175885
1: -0.6059399, 2.0574260, -0.7381821, 2.1788821, -2.7848220, 2.7956080
2: -1.3041267, 1.3373374, -1.5778151, 1.5577803, -2.8619070, 2.9151525
3: -0.9207745, 3.0303936, -1.0751319, 3.3567352, -4.2775097, 4.1055255
4: -1.7729893, 1.4975882, -2.0781775, 1.7465845, -3.5195737, 3.5757656

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5167794, 1.2480004, -1.7982037, 1.8027122
1: -0.6213336, 1.9295921, -0.6187482, 2.0452437, -2.6665773, 2.5483403
2: -1.3427572, 1.4046347, -1.3220568, 1.3385221, -2.6812792, 2.7266915
3: -0.9433498, 2.9237518, -0.9353175, 3.0362320, -3.9795818, 3.8590693
4: -1.7625704, 1.5530781, -1.7887945, 1.4996266, -3.2621970, 3.3418727

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5379963, 1.2854621, -1.7677467, 1.7608681
1: -0.5880079, 2.0136833, -0.6153898, 1.9354620, -2.5234699, 2.6290731
2: -1.2619104, 1.3153384, -1.3228397, 1.4002861, -2.6621966, 2.6381781
3: -0.8966851, 2.9580421, -0.9367595, 2.9233913, -3.8200765, 3.8948016
4: -1.7185078, 1.4698515, -1.7570276, 1.5475419, -3.2660496, 3.2268791

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5005965, 1.2284206, -1.7567037, 1.7813599
1: -0.6040082, 1.9247079, -0.6031599, 2.0058041, -2.6098123, 2.5278678
2: -1.3036880, 1.4001466, -1.2835169, 1.3217146, -2.6254027, 2.6836634
3: -0.9276228, 2.8863373, -0.9144611, 2.9694157, -3.8970385, 3.8007984
4: -1.7242384, 1.5467284, -1.7390747, 1.4786390, -3.2028775, 3.2858031

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5379963, 1.2854621, -1.8137453, 1.8187597
1: -0.6040082, 1.9247079, -0.6153898, 1.9354620, -2.5394702, 2.5400977
2: -1.3036880, 1.4001466, -1.3228397, 1.4002861, -2.7039742, 2.7229862
3: -0.9276228, 2.8863373, -0.9367595, 2.9233913, -3.8510141, 3.8230968
4: -1.7242384, 1.5467284, -1.7570276, 1.5475419, -3.2717803, 3.3037560

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5005965, 1.2284206, -1.7567037, 1.7813599
1: -0.6040082, 1.9247079, -0.6031599, 2.0058041, -2.6098123, 2.5278678
2: -1.3036880, 1.4001466, -1.2835169, 1.3217146, -2.6254027, 2.6836634
3: -0.9276228, 2.8863373, -0.9144611, 2.9694157, -3.8970385, 3.8007984
4: -1.7242384, 1.5467284, -1.7390747, 1.4786390, -3.2028775, 3.2858031

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.6025977, 1.3614380, -1.8897212, 1.8833611
1: -0.6040082, 1.9247079, -0.6731129, 2.0198145, -2.6238227, 2.5978208
2: -1.3036880, 1.4001466, -1.4475107, 1.4631717, -2.7668598, 2.8476572
3: -0.9276228, 2.8863373, -0.9972734, 3.1262207, -4.0538435, 3.8836107
4: -1.7242384, 1.5467284, -1.9038792, 1.6280352, -3.3522735, 3.4506075

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6025977, 1.3614380, -1.8437226, 1.8254695
1: -0.5880079, 2.0136833, -0.6731129, 2.0198145, -2.6078224, 2.6867962
2: -1.2619104, 1.3153384, -1.4475107, 1.4631717, -2.7250822, 2.7628491
3: -0.8966851, 2.9580421, -0.9972734, 3.1262207, -4.0229058, 3.9553156
4: -1.7185078, 1.4698515, -1.9038792, 1.6280352, -3.3465428, 3.3737307

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6025977, 1.3614380, -1.9116414, 1.8885305
1: -0.6213336, 1.9295921, -0.6731129, 2.0198145, -2.6411481, 2.6027050
2: -1.3427572, 1.4046347, -1.4475107, 1.4631717, -2.8059289, 2.8521454
3: -0.9433498, 2.9237518, -0.9972734, 3.1262207, -4.0695705, 3.9210253
4: -1.7625704, 1.5530781, -1.9038792, 1.6280352, -3.3906054, 3.4569573

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
time: 0.40 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 2.50 seconds
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9138581
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9139138
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9139067
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9129777, upper bound: 1.9140092
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9147138, upper bound: 1.9186916
IS_A1_A1_A1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9144503
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9064919, upper bound: 1.9146829
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
IS_A1_A1_A1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9193653
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9267035, upper bound: 1.9132031
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9229446, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9269233, upper bound: 1.9166133
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.50
Output dim: 0, lower bound: -1.9241591, upper bound: 1.9152623

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6215014, 1.4085987, -1.8786601, 1.9498835
1: -0.5353262, 1.9529831, -0.6910777, 2.1192222, -2.6545484, 2.6440609
2: -1.1786942, 1.4957325, -1.4793415, 1.5146111, -2.6933055, 2.9750741
3: -0.8506749, 2.7089930, -1.0249138, 3.2062330, -4.0569077, 3.7339067
4: -1.6099305, 1.6463493, -1.9659233, 1.6891631, -3.2990937, 3.6122727

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.36 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.6205978, 1.4072542, -1.8773156, 1.9489799
1: -0.5353262, 1.9529831, -0.6902347, 2.1172371, -2.6525633, 2.6432178
2: -1.1786942, 1.4957325, -1.4774981, 1.5133135, -2.6920078, 2.9732306
3: -0.8506749, 2.7089930, -1.0240054, 3.2029772, -4.0536518, 3.7329984
4: -1.6099305, 1.6463493, -1.9635334, 1.6875764, -3.2975068, 3.6098828

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9069216, upper bound: 1.9143787
time: 0.40 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.37 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9082590, upper bound: 1.9185962
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.7075887, 1.5035110, -1.9735724, 2.0359707
1: -0.5353262, 1.9529831, -0.7728319, 2.2361383, -2.7714646, 2.7258151
2: -1.1786942, 1.4957325, -1.6448193, 1.6279107, -2.8066049, 3.1405518
3: -0.8506749, 2.7089930, -1.1082888, 3.4591293, -4.3098040, 3.8172817
4: -1.6099305, 1.6463493, -2.1707525, 1.8386446, -3.4485750, 3.8171020

Time for backsubstitution: 1.50 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
time: 0.32 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
time: 0.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7438335, 1.5253004, -2.2143774, 2.2126141
1: -0.7561793, 2.1721611, -0.8104258, 2.3001223, -3.0563016, 2.9825869
2: -1.6216621, 1.5724510, -1.7379694, 1.6217314, -3.2433934, 3.3104205
3: -1.0972075, 3.4126759, -1.1569667, 3.6523972, -4.7496047, 4.5696425
4: -2.1124964, 1.7667136, -2.2834320, 1.8319958, -3.9444923, 4.0501456

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9282065, upper bound: 1.9179445
time: 0.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9194012
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.25 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9208493
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9156888, upper bound: 1.9204115
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9282065, upper bound: 1.9179445
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9194012, upper bound: 1.9194012

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.5925088, 1.3923821, -1.9286895, 1.9853988
1: -0.5818403, 2.0295284, -0.6679277, 2.1326437, -2.7144840, 2.6974561
2: -1.2697520, 1.5284014, -1.4484310, 1.4988259, -2.7685781, 2.9768324
3: -0.8890121, 2.7896709, -1.0020442, 3.1850576, -4.0740700, 3.7917151
4: -1.7242160, 1.6678995, -1.9370098, 1.6677970, -3.3920131, 3.6049094

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146858
time: 0.36 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.5925088, 1.3923821, -2.0211747, 2.0752230
1: -0.6841701, 2.1662390, -0.6679277, 2.1326437, -2.8168139, 2.8341668
2: -1.4798999, 1.6526315, -1.4484310, 1.4988259, -2.9787259, 3.1010625
3: -0.9925163, 3.1035147, -1.0020442, 3.1850576, -4.1775742, 4.1055589
4: -1.9907460, 1.8352203, -1.9370098, 1.6677970, -3.6585431, 3.7722301

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9156861, upper bound: 1.9160825
time: 0.34 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.6723809, 1.4565330, -2.1456099, 2.1411614
1: -0.7561793, 2.1721611, -0.7381821, 2.1788821, -2.9350615, 2.9103432
2: -1.6216621, 1.5724510, -1.5778151, 1.5577803, -3.1794424, 3.1502662
3: -1.0972075, 3.4126759, -1.0751319, 3.3567352, -4.4539428, 4.4878078
4: -2.1124964, 1.7667136, -2.0781775, 1.7465845, -3.8590808, 3.8448911

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9265877, upper bound: 1.9152623
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7575412, 1.5481710, -2.2372479, 2.2263217
1: -0.7561793, 2.1721611, -0.8227930, 2.2892652, -3.0454445, 2.9949541
2: -1.6216621, 1.5724510, -1.7448044, 1.6674051, -3.2890673, 3.3172555
3: -1.0972075, 3.4126759, -1.1639147, 3.6009045, -4.6981120, 4.5765905
4: -2.1124964, 1.7667136, -2.2804823, 1.8911014, -4.0035977, 4.0471959

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9136407, upper bound: 1.9155453
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9149990, upper bound: 1.9149989
time: 0.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.14 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
IS_A1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146858
IS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9156861, upper bound: 1.9160825
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9265877, upper bound: 1.9152623
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9136407, upper bound: 1.9155453
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9149990, upper bound: 1.9149989

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.5925088, 1.3923821, -1.8021541, 1.8612126
1: -0.4681202, 1.8429868, -0.6679277, 2.1326437, -2.6007638, 2.5109146
2: -1.0267184, 1.4059389, -1.4484310, 1.4988259, -2.5255442, 2.8543699
3: -0.7750952, 2.4403634, -1.0020442, 3.1850576, -3.9601529, 3.4424076
4: -1.4172378, 1.5203712, -1.9370098, 1.6677970, -3.0850348, 3.4573810

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2

### Relational analysis result of IS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9190137
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6161108, 1.3870221, -1.8876334, 1.8613183
1: -0.6059399, 2.0574260, -0.6884580, 2.0891037, -2.6950436, 2.7458839
2: -1.3041267, 1.3373374, -1.4694004, 1.4898863, -2.7940130, 2.8067379
3: -0.9207745, 3.0303936, -1.0221820, 3.1915951, -4.1123695, 4.0525756
4: -1.7729893, 1.4975882, -1.9522581, 1.6614250, -3.4344144, 3.4498463

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6191602, 1.3756136, -0.6596942, 1.4403439, -2.0595040, 2.0353079
1: -0.6878209, 2.0247331, -0.7270684, 2.1559572, -2.8437781, 2.7518015
2: -1.4858418, 1.4780548, -1.5553541, 1.5416453, -3.0274870, 3.0334089
3: -1.0179358, 3.1737194, -1.0614147, 3.3214741, -4.3394098, 4.2351341
4: -1.9349661, 1.6475326, -2.0495396, 1.7265241, -3.6614902, 3.6970720

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9265877, upper bound: 1.9152623
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9265877, upper bound: 1.9152623
time: 0.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.22 seconds
IS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
IS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9190137
IS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
IS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9265877, upper bound: 1.9152623
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.22
Output dim: 0, lower bound: -1.9265877, upper bound: 1.9152623

## BFS IS instance: IS_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.5925088, 1.3923821, -1.8021541, 1.8612126
1: -0.4681202, 1.8429868, -0.6679277, 2.1326437, -2.6007638, 2.5109146
2: -1.0267184, 1.4059389, -1.4484310, 1.4988259, -2.5255442, 2.8543699
3: -0.7750952, 2.4403634, -1.0020442, 3.1850576, -3.9601529, 3.4424076
4: -1.4172378, 1.5203712, -1.9370098, 1.6677970, -3.0850348, 3.4573810

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.5925088, 1.3923821, -1.8624436, 1.9208908
1: -0.5353262, 1.9529831, -0.6679277, 2.1326437, -2.6679699, 2.6209109
2: -1.1786942, 1.4957325, -1.4484310, 1.4988259, -2.6775203, 2.9441636
3: -0.8506749, 2.7089930, -1.0020442, 3.1850576, -4.0357323, 3.7110372
4: -1.6099305, 1.6463493, -1.9370098, 1.6677970, -3.2777276, 3.5833592

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9189668
time: 0.38 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9190137
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4991136, 1.3552921, -0.5925088, 1.3923821, -1.8914957, 1.9478009
1: -0.5415514, 1.9243438, -0.6679277, 2.1326437, -2.6741951, 2.5922716
2: -1.1696773, 1.4981003, -1.4484310, 1.4988259, -2.6685033, 2.9465313
3: -0.8462863, 2.6033816, -1.0020442, 3.1850576, -4.0313439, 3.6054258
4: -1.5904164, 1.6230114, -1.9370098, 1.6677970, -3.2582135, 3.5600212

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A1_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A1_A2

### Relational analysis result of IS_A1_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9160825
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5093265, 1.2443869, -1.7449982, 1.7545340
1: -0.6059399, 2.0574260, -0.6126885, 2.0398026, -2.6457424, 2.6701145
2: -1.3041267, 1.3373374, -1.3046255, 1.3357230, -2.6398497, 2.6419630
3: -0.9207745, 3.0303936, -0.9253378, 3.0163879, -3.9371624, 3.9557314
4: -1.7729893, 1.4975882, -1.7691002, 1.4960246, -3.2690139, 3.2666883

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.6711755, 1.4548311, -2.0225372, 1.9958172
1: -0.6357365, 1.9145675, -0.7370734, 2.1765194, -2.8122559, 2.6516409
2: -1.3678999, 1.4330735, -1.5753698, 1.5561364, -2.9240363, 3.0084434
3: -0.9600172, 2.9499054, -1.0738974, 3.3527908, -4.3128080, 4.0238028
4: -1.7817144, 1.5861579, -2.0752544, 1.7445345, -3.5262489, 3.6614122

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9149989
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9149990, upper bound: 1.9115793
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115881
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.6673212, 1.4338410, -0.6711755, 1.4548311, -2.1221523, 2.1050165
1: -0.7327099, 2.0694237, -0.7370734, 2.1765194, -2.9092293, 2.8064971
2: -1.5579519, 1.5491378, -1.5753698, 1.5561364, -3.1140883, 3.1245077
3: -1.0585232, 3.2334042, -1.0738974, 3.3527908, -4.4113140, 4.3073015
4: -2.0197039, 1.7408283, -2.0752544, 1.7445345, -3.7642384, 3.8160827

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9149989
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9149990, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115881
time: 0.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.07 seconds
IS_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
IS_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9195009
IS_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9189668
IS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9078991, upper bound: 1.9190137
IS_A1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
IS_A1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9160825
IS_A1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9149990, upper bound: 1.9115793
IS_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115881
IS_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9149990, upper bound: 1.9115793
IS_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.07
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9115881

## BFS IS instance: IS_A1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.4787123, 1.2552369, -1.6650088, 1.7474161
1: -0.4681202, 1.8429868, -0.5602863, 1.9525657, -2.4206858, 2.4032731
2: -1.0267184, 1.4059389, -1.2145548, 1.3622700, -2.3889885, 2.6204937
3: -0.7750952, 2.4403634, -0.8759546, 2.8103666, -3.5854619, 3.3163180
4: -1.4172378, 1.5203712, -1.6513958, 1.4909689, -2.9082067, 3.1717670

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144532
time: 0.33 seconds

## Relational analysis of IS_A1_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144503
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4097720, 1.2687038, -0.5094101, 1.2812526, -1.6910245, 1.7781140
1: -0.4681202, 1.8429868, -0.5916753, 1.9757643, -2.4438844, 2.4346621
2: -1.0267184, 1.4059389, -1.2868309, 1.3892144, -2.4159327, 2.6927698
3: -0.7750952, 2.4403634, -0.9098535, 2.9150295, -3.6901248, 3.3502169
4: -1.4172378, 1.5203712, -1.7333727, 1.5312274, -2.9484651, 3.2537439

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146858
time: 0.35 seconds

## Relational analysis of IS_A1_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146829
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.4787123, 1.2552369, -1.7252983, 1.8070943
1: -0.5353262, 1.9529831, -0.5602863, 1.9525657, -2.4878919, 2.5132694
2: -1.1786942, 1.4957325, -1.2145548, 1.3622700, -2.5409641, 2.7102873
3: -0.8506749, 2.7089930, -0.8759546, 2.8103666, -3.6610415, 3.5849476
4: -1.6099305, 1.6463493, -1.6513958, 1.4909689, -3.1008995, 3.2977452

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144503
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144503
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4700614, 1.3283820, -0.5094101, 1.2812526, -1.7513140, 1.8377922
1: -0.5353262, 1.9529831, -0.5916753, 1.9757643, -2.5110905, 2.5446584
2: -1.1786942, 1.4957325, -1.2868309, 1.3892144, -2.5679088, 2.7825634
3: -0.8506749, 2.7089930, -0.9098535, 2.9150295, -3.7657044, 3.6188464
4: -1.6099305, 1.6463493, -1.7333727, 1.5312274, -3.1411579, 3.3797221

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146829
time: 0.34 seconds

## Relational analysis of IS_A1_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146829
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5925088, 1.3923821, -1.7788434, 1.8267304
1: -0.4308859, 1.7372168, -0.6679277, 2.1326437, -2.5635295, 2.4051447
2: -0.9271202, 1.3788922, -1.4484310, 1.4988259, -2.4259462, 2.8273232
3: -0.7347761, 2.2486658, -1.0020442, 3.1850576, -3.9198337, 3.2507100
4: -1.2825458, 1.4811245, -1.9370098, 1.6677970, -2.9503427, 3.4181342

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5093265, 1.2443869, -1.7449982, 1.7545340
1: -0.6059399, 2.0574260, -0.6126885, 2.0398026, -2.6457424, 2.6701145
2: -1.3041267, 1.3373374, -1.3046255, 1.3357230, -2.6398497, 2.6419630
3: -0.9207745, 3.0303936, -0.9253378, 3.0163879, -3.9371624, 3.9557314
4: -1.7729893, 1.4975882, -1.7691002, 1.4960246, -3.2690139, 3.2666883

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.19 seconds
IS_A1_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144532
IS_A1_A1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144503
IS_A1_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146858
IS_A1_A1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146829
IS_A1_A1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144503
IS_A1_A1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9144503
IS_A1_A1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146829
IS_A1_A1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9061711, upper bound: 1.9146829
IS_A1_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
IS_A1_A2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
IS_A1_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623

## BFS IS instance: IS_A1_A2_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5925088, 1.3923821, -1.7788434, 1.8267304
1: -0.4308859, 1.7372168, -0.6679277, 2.1326437, -2.5635295, 2.4051447
2: -0.9271202, 1.3788922, -1.4484310, 1.4988259, -2.4259462, 2.8273232
3: -0.7347761, 2.2486658, -1.0020442, 3.1850576, -3.9198337, 3.2507100
4: -1.2825458, 1.4811245, -1.9370098, 1.6677970, -2.9503427, 3.4181342

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9189114
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.4363029, 1.2878690, -0.5925088, 1.3923821, -1.8286850, 1.8803778
1: -0.4974524, 1.8372626, -0.6679277, 2.1326437, -2.6300962, 2.5051904
2: -1.0805206, 1.4634794, -1.4484310, 1.4988259, -2.5793467, 2.9119105
3: -0.8067336, 2.5146861, -1.0020442, 3.1850576, -3.9917912, 3.5167303
4: -1.4770038, 1.6012537, -1.9370098, 1.6677970, -3.1448007, 3.5382636

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9175026
time: 0.34 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4991136, 1.3552921, -0.5925088, 1.3923821, -1.8914957, 1.9478009
1: -0.5415514, 1.9243438, -0.6679277, 2.1326437, -2.6741951, 2.5922716
2: -1.1696773, 1.4981003, -1.4484310, 1.4988259, -2.6685033, 2.9465313
3: -0.8462863, 2.6033816, -1.0020442, 3.1850576, -4.0313439, 3.6054258
4: -1.5904164, 1.6230114, -1.9370098, 1.6677970, -3.2582135, 3.5600212

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9160825
time: 0.39 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5093265, 1.2443869, -1.7449982, 1.7545340
1: -0.6059399, 2.0574260, -0.6126885, 2.0398026, -2.6457424, 2.6701145
2: -1.3041267, 1.3373374, -1.3046255, 1.3357230, -2.6398497, 2.6419630
3: -0.9207745, 3.0303936, -0.9253378, 3.0163879, -3.9371624, 3.9557314
4: -1.7729893, 1.4975882, -1.7691002, 1.4960246, -3.2690139, 3.2666883

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5597806, 1.2903037, -1.7909150, 1.8049881
1: -0.6059399, 2.0574260, -0.6338792, 1.9403448, -2.5462847, 2.6913052
2: -1.3041267, 1.3373374, -1.3613148, 1.4047705, -2.7088971, 2.6986523
3: -0.9207745, 3.0303936, -0.9532218, 2.9620132, -3.8827877, 3.9836154
4: -1.7729893, 1.4975882, -1.7948627, 1.5538219, -3.3268113, 3.2924509

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.31 seconds
IS_A1_A2_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9189114
IS_A1_A2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9175026
IS_A1_A2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
IS_A1_A2_A2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A2_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
IS_A1_A2_A2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9160825
IS_A1_A2_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.31
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623

## BFS IS instance: IS_A1_A2_A2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.4787123, 1.2552369, -1.6416981, 1.7129339
1: -0.4308859, 1.7372168, -0.5602863, 1.9525657, -2.3834515, 2.2975030
2: -0.9271202, 1.3788922, -1.2145548, 1.3622700, -2.2893901, 2.5934470
3: -0.7347761, 2.2486658, -0.8759546, 2.8103666, -3.5451427, 3.1246204
4: -1.2825458, 1.4811245, -1.6513958, 1.4909689, -2.7735147, 3.1325202

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127662
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5094101, 1.2812526, -1.6677139, 1.7436317
1: -0.4308859, 1.7372168, -0.5916753, 1.9757643, -2.4066501, 2.3288922
2: -0.9271202, 1.3788922, -1.2868309, 1.3892144, -2.3163347, 2.6657231
3: -0.7347761, 2.2486658, -0.9098535, 2.9150295, -3.6498055, 3.1585193
4: -1.2825458, 1.4811245, -1.7333727, 1.5312274, -2.8137732, 3.2144971

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4363029, 1.2878690, -0.4787123, 1.2552369, -1.6915398, 1.7665813
1: -0.4974524, 1.8372626, -0.5602863, 1.9525657, -2.4500182, 2.3975489
2: -1.0805206, 1.4634794, -1.2145548, 1.3622700, -2.4427905, 2.6780343
3: -0.8067336, 2.5146861, -0.8759546, 2.8103666, -3.6171002, 3.3906407
4: -1.4770038, 1.6012537, -1.6513958, 1.4909689, -2.9679728, 3.2526495

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
time: 0.34 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4363029, 1.2878690, -0.5094101, 1.2812526, -1.7175555, 1.7972791
1: -0.4974524, 1.8372626, -0.5916753, 1.9757643, -2.4732168, 2.4289379
2: -1.0805206, 1.4634794, -1.2868309, 1.3892144, -2.4697351, 2.7503104
3: -0.8067336, 2.5146861, -0.9098535, 2.9150295, -3.7217631, 3.4245396
4: -1.4770038, 1.6012537, -1.7333727, 1.5312274, -3.0082312, 3.3346264

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_A2_A1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5925088, 1.3923821, -1.7788434, 1.8267304
1: -0.4308859, 1.7372168, -0.6679277, 2.1326437, -2.5635295, 2.4051447
2: -0.9271202, 1.3788922, -1.4484310, 1.4988259, -2.4259462, 2.8273232
3: -0.7347761, 2.2486658, -1.0020442, 3.1850576, -3.9198337, 3.2507100
4: -1.2825458, 1.4811245, -1.9370098, 1.6677970, -2.9503427, 3.4181342

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5093265, 1.2443869, -1.7449982, 1.7545340
1: -0.6059399, 2.0574260, -0.6126885, 2.0398026, -2.6457424, 2.6701145
2: -1.3041267, 1.3373374, -1.3046255, 1.3357230, -2.6398497, 2.6419630
3: -0.9207745, 3.0303936, -0.9253378, 3.0163879, -3.9371624, 3.9557314
4: -1.7729893, 1.4975882, -1.7691002, 1.4960246, -3.2690139, 3.2666883

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5597806, 1.2903037, -1.7909150, 1.8049881
1: -0.6059399, 2.0574260, -0.6338792, 1.9403448, -2.5462847, 2.6913052
2: -1.3041267, 1.3373374, -1.3613148, 1.4047705, -2.7088971, 2.6986523
3: -0.9207745, 3.0303936, -0.9532218, 2.9620132, -3.8827877, 3.9836154
4: -1.7729893, 1.4975882, -1.7948627, 1.5538219, -3.3268113, 3.2924509

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5093265, 1.2443869, -1.7266715, 1.7321982
1: -0.5880079, 2.0136833, -0.6126885, 2.0398026, -2.6278105, 2.6263719
2: -1.2619104, 1.3153384, -1.3046255, 1.3357230, -2.5976334, 2.6199639
3: -0.8966851, 2.9580421, -0.9253378, 3.0163879, -3.9130731, 3.8833799
4: -1.7185078, 1.4698515, -1.7691002, 1.4960246, -3.2145324, 3.2389517

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5999899, 1.3601222, -1.8424067, 1.8228617
1: -0.5880079, 2.0136833, -0.6711960, 2.0180054, -2.6060133, 2.6848793
2: -1.2619104, 1.3153384, -1.4441347, 1.4615185, -2.7234290, 2.7594731
3: -0.8966851, 2.9580421, -0.9964314, 3.1247883, -4.0214734, 3.9544735
4: -1.7185078, 1.4698515, -1.9009428, 1.6262220, -3.3447299, 3.3707943

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.38 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.33 seconds
IS_A1_A2_A2_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127662
IS_A1_A2_A2_A1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
IS_A1_A2_A2_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
IS_A1_A2_A2_A1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
IS_A1_A2_A2_A1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
IS_A1_A2_A2_A1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
IS_A1_A2_A2_A1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
IS_A1_A2_A2_A1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
IS_A1_A2_A2_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
IS_A1_A2_A2_A2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
IS_A1_A2_A2_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.33
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5925088, 1.3923821, -1.7788434, 1.8267304
1: -0.4308859, 1.7372168, -0.6679277, 2.1326437, -2.5635295, 2.4051447
2: -0.9271202, 1.3788922, -1.4484310, 1.4988259, -2.4259462, 2.8273232
3: -0.7347761, 2.2486658, -1.0020442, 3.1850576, -3.9198337, 3.2507100
4: -1.2825458, 1.4811245, -1.9370098, 1.6677970, -2.9503427, 3.4181342

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9189114
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.4363029, 1.2878690, -0.5925088, 1.3923821, -1.8286850, 1.8803778
1: -0.4974524, 1.8372626, -0.6679277, 2.1326437, -2.6300962, 2.5051904
2: -1.0805206, 1.4634794, -1.4484310, 1.4988259, -2.5793467, 2.9119105
3: -0.8067336, 2.5146861, -1.0020442, 3.1850576, -3.9917912, 3.5167303
4: -1.4770038, 1.6012537, -1.9370098, 1.6677970, -3.1448007, 3.5382636

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9175026
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4991136, 1.3552921, -0.5925088, 1.3923821, -1.8914957, 1.9478009
1: -0.5415514, 1.9243438, -0.6679277, 2.1326437, -2.6741951, 2.5922716
2: -1.1696773, 1.4981003, -1.4484310, 1.4988259, -2.6685033, 2.9465313
3: -0.8462863, 2.6033816, -1.0020442, 3.1850576, -4.0313439, 3.6054258
4: -1.5904164, 1.6230114, -1.9370098, 1.6677970, -3.2582135, 3.5600212

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9160825
time: 0.39 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5093265, 1.2443869, -1.7449982, 1.7545340
1: -0.6059399, 2.0574260, -0.6126885, 2.0398026, -2.6457424, 2.6701145
2: -1.3041267, 1.3373374, -1.3046255, 1.3357230, -2.6398497, 2.6419630
3: -0.9207745, 3.0303936, -0.9253378, 3.0163879, -3.9371624, 3.9557314
4: -1.7729893, 1.4975882, -1.7691002, 1.4960246, -3.2690139, 3.2666883

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5597806, 1.2903037, -1.7909150, 1.8049881
1: -0.6059399, 2.0574260, -0.6338792, 1.9403448, -2.5462847, 2.6913052
2: -1.3041267, 1.3373374, -1.3613148, 1.4047705, -2.7088971, 2.6986523
3: -0.9207745, 3.0303936, -0.9532218, 2.9620132, -3.8827877, 3.9836154
4: -1.7729893, 1.4975882, -1.7948627, 1.5538219, -3.3268113, 3.2924509

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.4984221, 1.2241011, -1.7247124, 1.7436297
1: -0.6059399, 2.0574260, -0.6008134, 1.9953914, -2.6013312, 2.6582394
2: -1.3041267, 1.3373374, -1.2782278, 1.3162702, -2.6203971, 2.6155653
3: -0.9207745, 3.0303936, -0.9113355, 2.9574947, -3.8782692, 3.9417291
4: -1.7729893, 1.4975882, -1.7313890, 1.4713126, -3.2443018, 3.2289772

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5597806, 1.2903037, -1.7909150, 1.8049881
1: -0.6059399, 2.0574260, -0.6338792, 1.9403448, -2.5462847, 2.6913052
2: -1.3041267, 1.3373374, -1.3613148, 1.4047705, -2.7088971, 2.6986523
3: -0.9207745, 3.0303936, -0.9532218, 2.9620132, -3.8827877, 3.9836154
4: -1.7729893, 1.4975882, -1.7948627, 1.5538219, -3.3268113, 3.2924509

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.4984221, 1.2241011, -1.7743044, 1.7843549
1: -0.6213336, 1.9295921, -0.6008134, 1.9953914, -2.6167250, 2.5304055
2: -1.3427572, 1.4046347, -1.2782278, 1.3162702, -2.6590276, 2.6828625
3: -0.9433498, 2.9237518, -0.9113355, 2.9574947, -3.9008446, 3.8350873
4: -1.7625704, 1.5530781, -1.7313890, 1.4713126, -3.2338829, 3.2844672

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5093265, 1.2443869, -1.7266715, 1.7321982
1: -0.5880079, 2.0136833, -0.6126885, 2.0398026, -2.6278105, 2.6263719
2: -1.2619104, 1.3153384, -1.3046255, 1.3357230, -2.5976334, 2.6199639
3: -0.8966851, 2.9580421, -0.9253378, 3.0163879, -3.9130731, 3.8833799
4: -1.7185078, 1.4698515, -1.7691002, 1.4960246, -3.2145324, 3.2389517

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.4984221, 1.2241011, -1.7063856, 1.7212939
1: -0.5880079, 2.0136833, -0.6008134, 1.9953914, -2.5833993, 2.6144967
2: -1.2619104, 1.3153384, -1.2782278, 1.3162702, -2.5781808, 2.5935662
3: -0.8966851, 2.9580421, -0.9113355, 2.9574947, -3.8541799, 3.8693776
4: -1.7185078, 1.4698515, -1.7313890, 1.4713126, -3.1898203, 3.2012405

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.4984221, 1.2241011, -1.7743044, 1.7843549
1: -0.6213336, 1.9295921, -0.6008134, 1.9953914, -2.6167250, 2.5304055
2: -1.3427572, 1.4046347, -1.2782278, 1.3162702, -2.6590276, 2.6828625
3: -0.9433498, 2.9237518, -0.9113355, 2.9574947, -3.9008446, 3.8350873
4: -1.7625704, 1.5530781, -1.7313890, 1.4713126, -3.2338829, 3.2844672

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5379963, 1.2854621, -1.7860734, 1.7832038
1: -0.6059399, 2.0574260, -0.6153898, 1.9354620, -2.5414019, 2.6728158
2: -1.3041267, 1.3373374, -1.3228397, 1.4002861, -2.7044129, 2.6601772
3: -0.9207745, 3.0303936, -0.9367595, 2.9233913, -3.8441658, 3.9671531
4: -1.7729893, 1.4975882, -1.7570276, 1.5475419, -3.3205311, 3.2546158

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5379963, 1.2854621, -1.8356655, 1.8239291
1: -0.6213336, 1.9295921, -0.6153898, 1.9354620, -2.5567956, 2.5449820
2: -1.3427572, 1.4046347, -1.3228397, 1.4002861, -2.7430434, 2.7274745
3: -0.9433498, 2.9237518, -0.9367595, 2.9233913, -3.8667412, 3.8605113
4: -1.7625704, 1.5530781, -1.7570276, 1.5475419, -3.3101122, 3.3101058

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5379963, 1.2854621, -1.7677467, 1.7608681
1: -0.5880079, 2.0136833, -0.6153898, 1.9354620, -2.5234699, 2.6290731
2: -1.2619104, 1.3153384, -1.3228397, 1.4002861, -2.6621966, 2.6381781
3: -0.8966851, 2.9580421, -0.9367595, 2.9233913, -3.8200765, 3.8948016
4: -1.7185078, 1.4698515, -1.7570276, 1.5475419, -3.2660496, 3.2268791

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5379963, 1.2854621, -1.8356655, 1.8239291
1: -0.6213336, 1.9295921, -0.6153898, 1.9354620, -2.5567956, 2.5449820
2: -1.3427572, 1.4046347, -1.3228397, 1.4002861, -2.7430434, 2.7274745
3: -0.9433498, 2.9237518, -0.9367595, 2.9233913, -3.8667412, 3.8605113
4: -1.7625704, 1.5530781, -1.7570276, 1.5475419, -3.3101122, 3.3101058

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5999899, 1.3601222, -1.8424067, 1.8228617
1: -0.5880079, 2.0136833, -0.6711960, 2.0180054, -2.6060133, 2.6848793
2: -1.2619104, 1.3153384, -1.4441347, 1.4615185, -2.7234290, 2.7594731
3: -0.8966851, 2.9580421, -0.9964314, 3.1247883, -4.0214734, 3.9544735
4: -1.7185078, 1.4698515, -1.9009428, 1.6262220, -3.3447299, 3.3707943

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
time: 0.39 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.60 seconds
IS_A1_A2_A2_A2_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9189114
IS_A1_A2_A2_A2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A2_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9175026
IS_A1_A2_A2_A2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
IS_A1_A2_A2_A2_A2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
IS_A1_A2_A2_A2_A2_A2_A2_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
IS_A1_A2_A2_A2_A2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9160825
IS_A1_A2_A2_A2_A2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
IS_A2_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
IS_A2_B1_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9130443
IS_A2_B1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9251727, upper bound: 1.9156885
IS_A2_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.60
Output dim: 0, lower bound: -1.9233023, upper bound: 1.9152623

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.4787123, 1.2552369, -1.6416981, 1.7129339
1: -0.4308859, 1.7372168, -0.5602863, 1.9525657, -2.3834515, 2.2975030
2: -0.9271202, 1.3788922, -1.2145548, 1.3622700, -2.2893901, 2.5934470
3: -0.7347761, 2.2486658, -0.8759546, 2.8103666, -3.5451427, 3.1246204
4: -1.2825458, 1.4811245, -1.6513958, 1.4909689, -2.7735147, 3.1325202

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127662
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5094101, 1.2812526, -1.6677139, 1.7436317
1: -0.4308859, 1.7372168, -0.5916753, 1.9757643, -2.4066501, 2.3288922
2: -0.9271202, 1.3788922, -1.2868309, 1.3892144, -2.3163347, 2.6657231
3: -0.7347761, 2.2486658, -0.9098535, 2.9150295, -3.6498055, 3.1585193
4: -1.2825458, 1.4811245, -1.7333727, 1.5312274, -2.8137732, 3.2144971

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138781
time: 0.40 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4363029, 1.2878690, -0.4787123, 1.2552369, -1.6915398, 1.7665813
1: -0.4974524, 1.8372626, -0.5602863, 1.9525657, -2.4500182, 2.3975489
2: -1.0805206, 1.4634794, -1.2145548, 1.3622700, -2.4427905, 2.6780343
3: -0.8067336, 2.5146861, -0.8759546, 2.8103666, -3.6171002, 3.3906407
4: -1.4770038, 1.6012537, -1.6513958, 1.4909689, -2.9679728, 3.2526495

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9127497
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4363029, 1.2878690, -0.5094101, 1.2812526, -1.7175555, 1.7972791
1: -0.4974524, 1.8372626, -0.5916753, 1.9757643, -2.4732168, 2.4289379
2: -1.0805206, 1.4634794, -1.2868309, 1.3892144, -2.4697351, 2.7503104
3: -0.8067336, 2.5146861, -0.9098535, 2.9150295, -3.7217631, 3.4245396
4: -1.4770038, 1.6012537, -1.7333727, 1.5312274, -3.0082312, 3.3346264

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9061490, upper bound: 1.9138664
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3864612, 1.2342216, -0.5925088, 1.3923821, -1.7788434, 1.8267304
1: -0.4308859, 1.7372168, -0.6679277, 2.1326437, -2.5635295, 2.4051447
2: -0.9271202, 1.3788922, -1.4484310, 1.4988259, -2.4259462, 2.8273232
3: -0.7347761, 2.2486658, -1.0020442, 3.1850576, -3.9198337, 3.2507100
4: -1.2825458, 1.4811245, -1.9370098, 1.6677970, -2.9503427, 3.4181342

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9193448
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9078769, upper bound: 1.9186207
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_A2_A2_A2_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5925088, 1.3923821, -1.9805964, 2.0351474
1: -0.6440235, 2.0589573, -0.6679277, 2.1326437, -2.7766671, 2.7268851
2: -1.3809800, 1.6204529, -1.4484310, 1.4988259, -2.8798060, 3.0688839
3: -0.9490790, 2.9139991, -1.0020442, 3.1850576, -4.1341367, 3.9160433
4: -1.8595581, 1.7889128, -1.9370098, 1.6677970, -3.5273552, 3.7259226

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9196002
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A2_A2_A2_A2_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2_A2_A2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9144053, upper bound: 1.9196002
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5093265, 1.2443869, -1.7449982, 1.7545340
1: -0.6059399, 2.0574260, -0.6126885, 2.0398026, -2.6457424, 2.6701145
2: -1.3041267, 1.3373374, -1.3046255, 1.3357230, -2.6398497, 2.6419630
3: -0.9207745, 3.0303936, -0.9253378, 3.0163879, -3.9371624, 3.9557314
4: -1.7729893, 1.4975882, -1.7691002, 1.4960246, -3.2690139, 3.2666883

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5597806, 1.2903037, -1.7909150, 1.8049881
1: -0.6059399, 2.0574260, -0.6338792, 1.9403448, -2.5462847, 2.6913052
2: -1.3041267, 1.3373374, -1.3613148, 1.4047705, -2.7088971, 2.6986523
3: -0.9207745, 3.0303936, -0.9532218, 2.9620132, -3.8827877, 3.9836154
4: -1.7729893, 1.4975882, -1.7948627, 1.5538219, -3.3268113, 3.2924509

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5093265, 1.2443869, -1.7266715, 1.7321982
1: -0.5880079, 2.0136833, -0.6126885, 2.0398026, -2.6278105, 2.6263719
2: -1.2619104, 1.3153384, -1.3046255, 1.3357230, -2.5976334, 2.6199639
3: -0.8966851, 2.9580421, -0.9253378, 3.0163879, -3.9130731, 3.8833799
4: -1.7185078, 1.4698515, -1.7691002, 1.4960246, -3.2145324, 3.2389517

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.4984221, 1.2241011, -1.7247124, 1.7436297
1: -0.6059399, 2.0574260, -0.6008134, 1.9953914, -2.6013312, 2.6582394
2: -1.3041267, 1.3373374, -1.2782278, 1.3162702, -2.6203971, 2.6155653
3: -0.9207745, 3.0303936, -0.9113355, 2.9574947, -3.8782692, 3.9417291
4: -1.7729893, 1.4975882, -1.7313890, 1.4713126, -3.2443018, 3.2289772

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.4984221, 1.2241011, -1.7743044, 1.7843549
1: -0.6213336, 1.9295921, -0.6008134, 1.9953914, -2.6167250, 2.5304055
2: -1.3427572, 1.4046347, -1.2782278, 1.3162702, -2.6590276, 2.6828625
3: -0.9433498, 2.9237518, -0.9113355, 2.9574947, -3.9008446, 3.8350873
4: -1.7625704, 1.5530781, -1.7313890, 1.4713126, -3.2338829, 3.2844672

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5597806, 1.2903037, -1.7909150, 1.8049881
1: -0.6059399, 2.0574260, -0.6338792, 1.9403448, -2.5462847, 2.6913052
2: -1.3041267, 1.3373374, -1.3613148, 1.4047705, -2.7088971, 2.6986523
3: -0.9207745, 3.0303936, -0.9532218, 2.9620132, -3.8827877, 3.9836154
4: -1.7729893, 1.4975882, -1.7948627, 1.5538219, -3.3268113, 3.2924509

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.4984221, 1.2241011, -1.7063856, 1.7212939
1: -0.5880079, 2.0136833, -0.6008134, 1.9953914, -2.5833993, 2.6144967
2: -1.2619104, 1.3153384, -1.2782278, 1.3162702, -2.5781808, 2.5935662
3: -0.8966851, 2.9580421, -0.9113355, 2.9574947, -3.8541799, 3.8693776
4: -1.7185078, 1.4698515, -1.7313890, 1.4713126, -3.1898203, 3.2012405

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.4984221, 1.2241011, -1.7743044, 1.7843549
1: -0.6213336, 1.9295921, -0.6008134, 1.9953914, -2.6167250, 2.5304055
2: -1.3427572, 1.4046347, -1.2782278, 1.3162702, -2.6590276, 2.6828625
3: -0.9433498, 2.9237518, -0.9113355, 2.9574947, -3.9008446, 3.8350873
4: -1.7625704, 1.5530781, -1.7313890, 1.4713126, -3.2338829, 3.2844672

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5093265, 1.2443869, -1.7266715, 1.7321982
1: -0.5880079, 2.0136833, -0.6126885, 2.0398026, -2.6278105, 2.6263719
2: -1.2619104, 1.3153384, -1.3046255, 1.3357230, -2.5976334, 2.6199639
3: -0.8966851, 2.9580421, -0.9253378, 3.0163879, -3.9130731, 3.8833799
4: -1.7185078, 1.4698515, -1.7691002, 1.4960246, -3.2145324, 3.2389517

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5093265, 1.2443869, -1.7726700, 1.7900898
1: -0.6040082, 1.9247079, -0.6126885, 2.0398026, -2.6438107, 2.5373964
2: -1.3036880, 1.4001466, -1.3046255, 1.3357230, -2.6394110, 2.7047720
3: -0.9276228, 2.8863373, -0.9253378, 3.0163879, -3.9440107, 3.8116751
4: -1.7242384, 1.5467284, -1.7691002, 1.4960246, -3.2202630, 3.3158286

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5597806, 1.2903037, -1.8185868, 1.8405440
1: -0.6040082, 1.9247079, -0.6338792, 1.9403448, -2.5443530, 2.5585871
2: -1.3036880, 1.4001466, -1.3613148, 1.4047705, -2.7084584, 2.7614613
3: -0.9276228, 2.8863373, -0.9532218, 2.9620132, -3.8896360, 3.8395591
4: -1.7242384, 1.5467284, -1.7948627, 1.5538219, -3.2780604, 3.3415911

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5093265, 1.2443869, -1.7266715, 1.7321982
1: -0.5880079, 2.0136833, -0.6126885, 2.0398026, -2.6278105, 2.6263719
2: -1.2619104, 1.3153384, -1.3046255, 1.3357230, -2.5976334, 2.6199639
3: -0.8966851, 2.9580421, -0.9253378, 3.0163879, -3.9130731, 3.8833799
4: -1.7185078, 1.4698515, -1.7691002, 1.4960246, -3.2145324, 3.2389517

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5093265, 1.2443869, -1.7945902, 1.7952592
1: -0.6213336, 1.9295921, -0.6126885, 2.0398026, -2.6611362, 2.5422807
2: -1.3427572, 1.4046347, -1.3046255, 1.3357230, -2.6784801, 2.7092602
3: -0.9433498, 2.9237518, -0.9253378, 3.0163879, -3.9597378, 3.8490896
4: -1.7625704, 1.5530781, -1.7691002, 1.4960246, -3.2585950, 3.3221784

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.4984221, 1.2241011, -1.7063856, 1.7212939
1: -0.5880079, 2.0136833, -0.6008134, 1.9953914, -2.5833993, 2.6144967
2: -1.2619104, 1.3153384, -1.2782278, 1.3162702, -2.5781808, 2.5935662
3: -0.8966851, 2.9580421, -0.9113355, 2.9574947, -3.8541799, 3.8693776
4: -1.7185078, 1.4698515, -1.7313890, 1.4713126, -3.1898203, 3.2012405

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.4984221, 1.2241011, -1.7523842, 1.7791855
1: -0.6040082, 1.9247079, -0.6008134, 1.9953914, -2.5993996, 2.5255213
2: -1.3036880, 1.4001466, -1.2782278, 1.3162702, -2.6199584, 2.6783743
3: -0.9276228, 2.8863373, -0.9113355, 2.9574947, -3.8851175, 3.7976727
4: -1.7242384, 1.5467284, -1.7313890, 1.4713126, -3.1955509, 3.2781174

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5597806, 1.2903037, -1.8185868, 1.8405440
1: -0.6040082, 1.9247079, -0.6338792, 1.9403448, -2.5443530, 2.5585871
2: -1.3036880, 1.4001466, -1.3613148, 1.4047705, -2.7084584, 2.7614613
3: -0.9276228, 2.8863373, -0.9532218, 2.9620132, -3.8896360, 3.8395591
4: -1.7242384, 1.5467284, -1.7948627, 1.5538219, -3.2780604, 3.3415911

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.4984221, 1.2241011, -1.7063856, 1.7212939
1: -0.5880079, 2.0136833, -0.6008134, 1.9953914, -2.5833993, 2.6144967
2: -1.2619104, 1.3153384, -1.2782278, 1.3162702, -2.5781808, 2.5935662
3: -0.8966851, 2.9580421, -0.9113355, 2.9574947, -3.8541799, 3.8693776
4: -1.7185078, 1.4698515, -1.7313890, 1.4713126, -3.1898203, 3.2012405

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.4984221, 1.2241011, -1.7743044, 1.7843549
1: -0.6213336, 1.9295921, -0.6008134, 1.9953914, -2.6167250, 2.5304055
2: -1.3427572, 1.4046347, -1.2782278, 1.3162702, -2.6590276, 2.6828625
3: -0.9433498, 2.9237518, -0.9113355, 2.9574947, -3.9008446, 3.8350873
4: -1.7625704, 1.5530781, -1.7313890, 1.4713126, -3.2338829, 3.2844672

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5597806, 1.2903037, -1.7725883, 1.7826524
1: -0.5880079, 2.0136833, -0.6338792, 1.9403448, -2.5283527, 2.6475625
2: -1.2619104, 1.3153384, -1.3613148, 1.4047705, -2.6666808, 2.6766531
3: -0.8966851, 2.9580421, -0.9532218, 2.9620132, -3.8586984, 3.9112639
4: -1.7185078, 1.4698515, -1.7948627, 1.5538219, -3.2723298, 3.2647142

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5597806, 1.2903037, -1.8405070, 1.8457134
1: -0.6213336, 1.9295921, -0.6338792, 1.9403448, -2.5616784, 2.5634713
2: -1.3427572, 1.4046347, -1.3613148, 1.4047705, -2.7475276, 2.7659495
3: -0.9433498, 2.9237518, -0.9532218, 2.9620132, -3.9053631, 3.8769736
4: -1.7625704, 1.5530781, -1.7948627, 1.5538219, -3.3163924, 3.3479409

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5379963, 1.2854621, -1.7860734, 1.7832038
1: -0.6059399, 2.0574260, -0.6153898, 1.9354620, -2.5414019, 2.6728158
2: -1.3041267, 1.3373374, -1.3228397, 1.4002861, -2.7044129, 2.6601772
3: -0.9207745, 3.0303936, -0.9367595, 2.9233913, -3.8441658, 3.9671531
4: -1.7729893, 1.4975882, -1.7570276, 1.5475419, -3.3205311, 3.2546158

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5379963, 1.2854621, -1.8356655, 1.8239291
1: -0.6213336, 1.9295921, -0.6153898, 1.9354620, -2.5567956, 2.5449820
2: -1.3427572, 1.4046347, -1.3228397, 1.4002861, -2.7430434, 2.7274745
3: -0.9433498, 2.9237518, -0.9367595, 2.9233913, -3.8667412, 3.8605113
4: -1.7625704, 1.5530781, -1.7570276, 1.5475419, -3.3101122, 3.3101058

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5379963, 1.2854621, -1.7677467, 1.7608681
1: -0.5880079, 2.0136833, -0.6153898, 1.9354620, -2.5234699, 2.6290731
2: -1.2619104, 1.3153384, -1.3228397, 1.4002861, -2.6621966, 2.6381781
3: -0.8966851, 2.9580421, -0.9367595, 2.9233913, -3.8200765, 3.8948016
4: -1.7185078, 1.4698515, -1.7570276, 1.5475419, -3.2660496, 3.2268791

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5379963, 1.2854621, -1.8356655, 1.8239291
1: -0.6213336, 1.9295921, -0.6153898, 1.9354620, -2.5567956, 2.5449820
2: -1.3427572, 1.4046347, -1.3228397, 1.4002861, -2.7430434, 2.7274745
3: -0.9433498, 2.9237518, -0.9367595, 2.9233913, -3.8667412, 3.8605113
4: -1.7625704, 1.5530781, -1.7570276, 1.5475419, -3.3101122, 3.3101058

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5005965, 1.2284206, -1.7786239, 1.7865293
1: -0.6213336, 1.9295921, -0.6031599, 2.0058041, -2.6271377, 2.5327520
2: -1.3427572, 1.4046347, -1.2835169, 1.3217146, -2.6644719, 2.6881516
3: -0.9433498, 2.9237518, -0.9144611, 2.9694157, -3.9127655, 3.8382130
4: -1.7625704, 1.5530781, -1.7390747, 1.4786390, -3.2412095, 3.2921529

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9126036
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9115793
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227748, upper bound: 1.9152623
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5005965, 1.2284206, -1.7107052, 1.7234683
1: -0.5880079, 2.0136833, -0.6031599, 2.0058041, -2.5938120, 2.6168432
2: -1.2619104, 1.3153384, -1.2835169, 1.3217146, -2.5836251, 2.5988553
3: -0.8966851, 2.9580421, -0.9144611, 2.9694157, -3.8661008, 3.8725033
4: -1.7185078, 1.4698515, -1.7390747, 1.4786390, -3.1971469, 3.2089262

Time for backsubstitution: 1.68 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=2.269134044647217
rel_dist={0: [-1.9301230654393355, 1.9301230654393358]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178776, upper bound: 1.9201370
time: 0.33 seconds

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

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190335
time: 0.33 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9148831, upper bound: 1.9189183
time: 0.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.6890769, 1.4687806, -0.7438335, 1.5253004, -2.2143774, 2.2126141
1: -0.7561793, 2.1721611, -0.8104258, 2.3001223, -3.0563016, 2.9825869
2: -1.6216621, 1.5724510, -1.7379694, 1.6217314, -3.2433934, 3.3104205
3: -1.0972075, 3.4126759, -1.1569667, 3.6523972, -4.7496047, 4.5696425
4: -2.1124964, 1.7667136, -2.2834320, 1.8319958, -3.9444923, 4.0501456

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
time: 0.40 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9264286, upper bound: 1.9264285
time: 0.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.30 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.9099164, upper bound: 1.9190335
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.9148831, upper bound: 1.9189183
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.9264286, upper bound: 1.9264285

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.5363073, 1.3928900, -0.5508800, 1.3501766, -1.8864839, 1.9437699
1: -0.5818403, 2.0295284, -0.6288738, 2.0743403, -2.6561806, 2.6584022
2: -1.2697520, 1.5284014, -1.3720207, 1.4563724, -2.7261243, 2.9004221
3: -0.8890121, 2.7896709, -0.9553370, 3.0639458, -3.9529579, 3.7450080
4: -1.7242160, 1.6678995, -1.8440752, 1.6113482, -3.3355641, 3.5119748

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070537, upper bound: 1.9170146
time: 0.31 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9053106, upper bound: 1.9143174
time: 0.36 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.6287925, 1.4827141, -0.5464194, 1.3435769, -1.9723694, 2.0291333
1: -0.6841701, 2.1662390, -0.6242437, 2.0614638, -2.7456341, 2.7904828
2: -1.4798999, 1.6526315, -1.3620682, 1.4495068, -2.9294066, 3.0146997
3: -0.9925163, 3.1035147, -0.9499125, 3.0446405, -4.0371571, 4.0534272
4: -1.9907460, 1.8352203, -1.8304243, 1.6020581, -3.5928040, 3.6656446

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9148783, upper bound: 1.9157927
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9143932, upper bound: 1.9186078
time: 0.38 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6326952, 1.3966057, -1.8972170, 1.8779027
1: -0.6059399, 2.0574260, -0.7104359, 2.1396341, -2.7455740, 2.7678618
2: -1.3041267, 1.3373374, -1.5290766, 1.4976027, -2.8017294, 2.8664141
3: -0.9207745, 3.0303936, -1.0479927, 3.3187103, -4.2394848, 4.0783863
4: -1.7729893, 1.4975882, -2.0334187, 1.6749433, -3.4479327, 3.5310068

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.38 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
time: 0.37 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.6191602, 1.3756136, -0.7110205, 1.4855890, -2.1047492, 2.0866342
1: -0.6878209, 2.0247331, -0.7812600, 2.2454538, -2.9332747, 2.8059931
2: -1.4858418, 1.4780548, -1.6804743, 1.5818183, -3.0676601, 3.1585290
3: -1.0179358, 3.1737194, -1.1210933, 3.5633526, -4.5812883, 4.2948127
4: -1.9349661, 1.6475326, -2.2109051, 1.7821966, -3.7171626, 3.8584375

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9228623
time: 0.39 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9147438
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.18 seconds
IS_A1_A1_A1, status: Status.VERIFIED, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9070537, upper bound: 1.9170146
IS_A1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9053106, upper bound: 1.9143174
IS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9148783, upper bound: 1.9157927
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9143932, upper bound: 1.9186078
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9228623
IS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 2.18
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9147438

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5508800, 1.3501766, -1.9383909, 1.9935186
1: -0.6440235, 2.0589573, -0.6288738, 2.0743403, -2.7183638, 2.6878312
2: -1.3809800, 1.6204529, -1.3720207, 1.4563724, -2.8373523, 2.9924736
3: -0.9490790, 2.9139991, -0.9553370, 3.0639458, -4.0130248, 3.8693361
4: -1.8595581, 1.7889128, -1.8440752, 1.6113482, -3.4709063, 3.6329880

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9186078
time: 0.34 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9143932, upper bound: 1.9186078
time: 0.38 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5306449, 1.2748586, -1.7754699, 1.7758524
1: -0.6059399, 2.0574260, -0.6343837, 2.1120329, -2.7179728, 2.6918097
2: -1.3041267, 1.3373374, -1.3630619, 1.3633302, -2.6674571, 2.7003994
3: -0.9207745, 3.0303936, -0.9513988, 3.1380749, -4.0588493, 3.9817924
4: -1.7729893, 1.4975882, -1.8497353, 1.5318481, -3.3048372, 3.3473234

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.38 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6604972, 1.4221686, -1.9227799, 1.9057047
1: -0.6059399, 2.0574260, -0.7327771, 2.1478353, -2.7537751, 2.7902031
2: -1.3041267, 1.3373374, -1.5855274, 1.5161924, -2.8203192, 2.9228649
3: -0.9207745, 3.0303936, -1.0668936, 3.4070873, -4.3278618, 4.0972872
4: -1.7729893, 1.4975882, -2.0916843, 1.7000986, -3.4730878, 3.5892725

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
time: 0.38 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.7426500, 1.5236474, -2.0913534, 2.0672917
1: -0.6357365, 1.9145675, -0.8093333, 2.2978258, -2.9335623, 2.7239008
2: -1.3678999, 1.4330735, -1.7356033, 1.6201339, -2.9880338, 3.1686769
3: -0.9600172, 2.9499054, -1.1557422, 3.6485500, -4.6085672, 4.1056476
4: -1.7817144, 1.5861579, -2.2806015, 1.8300055, -3.6117201, 3.8667593

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.36 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9228623
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.20 seconds
IS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9099137, upper bound: 1.9186078
IS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9143932, upper bound: 1.9186078
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9228623

## BFS IS instance: IS_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4991136, 1.3552921, -0.5508800, 1.3501766, -1.8492901, 1.9061720
1: -0.5415514, 1.9243438, -0.6288738, 2.0743403, -2.6158917, 2.5532176
2: -1.1696773, 1.4981003, -1.3720207, 1.4563724, -2.6260495, 2.8701210
3: -0.8462863, 2.6033816, -0.9553370, 3.0639458, -3.9102321, 3.5587187
4: -1.5904164, 1.6230114, -1.8440752, 1.6113482, -3.2017646, 3.4670866

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of IS_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A1_A1

### Relational analysis result of IS_A1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070492, upper bound: 1.9168716
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_A1_A2

### Relational analysis result of IS_A1_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9053106, upper bound: 1.9135984
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5882143, 1.4426386, -0.5464194, 1.3435769, -1.9317912, 1.9890580
1: -0.6440235, 2.0589573, -0.6242437, 2.0614638, -2.7054873, 2.6832011
2: -1.3809800, 1.6204529, -1.3620682, 1.4495068, -2.8304868, 2.9825211
3: -0.9490790, 2.9139991, -0.9499125, 3.0446405, -3.9937196, 3.8639116
4: -1.8595581, 1.7889128, -1.8304243, 1.6020581, -3.4616160, 3.6193371

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9127651, upper bound: 1.9123148
time: 0.41 seconds

## Relational analysis of IS_A1_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108021, upper bound: 1.9137614
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5306449, 1.2748586, -1.7754699, 1.7758524
1: -0.6059399, 2.0574260, -0.6343837, 2.1120329, -2.7179728, 2.6918097
2: -1.3041267, 1.3373374, -1.3630619, 1.3633302, -2.6674571, 2.7003994
3: -0.9207745, 3.0303936, -0.9513988, 3.1380749, -4.0588493, 3.9817924
4: -1.7729893, 1.4975882, -1.8497353, 1.5318481, -3.3048372, 3.3473234

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.6604972, 1.4221686, -1.9227799, 1.9057047
1: -0.6059399, 2.0574260, -0.7327771, 2.1478353, -2.7537751, 2.7902031
2: -1.3041267, 1.3373374, -1.5855274, 1.5161924, -2.8203192, 2.9228649
3: -0.9207745, 3.0303936, -1.0668936, 3.4070873, -4.3278618, 4.0972872
4: -1.7729893, 1.4975882, -2.0916843, 1.7000986, -3.4730878, 3.5892725

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116005, upper bound: 1.9147438
time: 0.35 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6604972, 1.4221686, -1.9723719, 1.9464300
1: -0.6213336, 1.9295921, -0.7327771, 2.1478353, -2.7691689, 2.6623693
2: -1.3427572, 1.4046347, -1.5855274, 1.5161924, -2.8589497, 2.9901621
3: -0.9433498, 2.9237518, -1.0668936, 3.4070873, -4.3504372, 3.9906454
4: -1.7625704, 1.5530781, -2.0916843, 1.7000986, -3.4626689, 3.6447625

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.6320639, 1.3956220, -1.8623216, 1.8293648
1: -0.5722542, 1.9478822, -0.7098413, 2.1381526, -2.7104068, 2.6577234
2: -1.2166643, 1.2926129, -1.5277510, 1.4966727, -2.7133369, 2.8203640
3: -0.8774633, 2.8582191, -1.0473237, 3.3162251, -4.1936884, 3.9055429
4: -1.6534157, 1.4405326, -2.0316648, 1.6737920, -3.3272076, 3.4721975

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.38 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.7110205, 1.4855890, -2.0532951, 2.0356622
1: -0.6357365, 1.9145675, -0.7812600, 2.2454538, -2.8811903, 2.6958275
2: -1.3678999, 1.4330735, -1.6804743, 1.5818183, -2.9497182, 3.1135478
3: -0.9600172, 2.9499054, -1.1210933, 3.5633526, -4.5233698, 4.0709987
4: -1.7817144, 1.5861579, -2.2109051, 1.7821966, -3.5639110, 3.7970629

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.37 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623
time: 0.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.25 seconds
IS_A1_A2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9070492, upper bound: 1.9168716
IS_A1_A2_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9053106, upper bound: 1.9135984
IS_A1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9127651, upper bound: 1.9123148
IS_A1_A2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9108021, upper bound: 1.9137614
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
IS_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9116005, upper bound: 1.9147438
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
IS_A2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.25
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5306449, 1.2748586, -1.7754699, 1.7758524
1: -0.6059399, 2.0574260, -0.6343837, 2.1120329, -2.7179728, 2.6918097
2: -1.3041267, 1.3373374, -1.3630619, 1.3633302, -2.6674571, 2.7003994
3: -0.9207745, 3.0303936, -0.9513988, 3.1380749, -4.0588493, 3.9817924
4: -1.7729893, 1.4975882, -1.8497353, 1.5318481, -3.3048372, 3.3473234

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5837932, 1.3274705, -1.8280818, 1.8290007
1: -0.6059399, 2.0574260, -0.6598225, 2.0440698, -2.6500096, 2.7172484
2: -1.3041267, 1.3373374, -1.4269304, 1.4358747, -2.7400014, 2.7642679
3: -0.9207745, 3.0303936, -0.9843640, 3.1207066, -4.0414810, 4.0147576
4: -1.7729893, 1.4975882, -1.8964205, 1.5964544, -3.3694437, 3.3940086

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9208025, upper bound: 1.9126412
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9208025, upper bound: 1.9147438
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5218196, 1.2599968, -1.8102001, 1.8077524
1: -0.6213336, 1.9295921, -0.6245918, 2.0851417, -2.7064753, 2.5541840
2: -1.3427572, 1.4046347, -1.3428059, 1.3493736, -2.6921308, 2.7474406
3: -0.9433498, 2.9237518, -0.9403701, 3.0976830, -4.0410328, 3.8641219
4: -1.7625704, 1.5530781, -1.8229065, 1.5146019, -3.2771723, 3.3759847

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6604972, 1.4221686, -1.9723719, 1.9464300
1: -0.6213336, 1.9295921, -0.7327771, 2.1478353, -2.7691689, 2.6623693
2: -1.3427572, 1.4046347, -1.5855274, 1.5161924, -2.8589497, 2.9901621
3: -0.9433498, 2.9237518, -1.0668936, 3.4070873, -4.3504372, 3.9906454
4: -1.7625704, 1.5530781, -2.0916843, 1.7000986, -3.4626689, 3.6447625

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5302949, 1.2742450, -1.7409446, 1.7275958
1: -0.5722542, 1.9478822, -0.6340547, 2.1110892, -2.6833434, 2.5819368
2: -1.2166643, 1.2926129, -1.3623152, 1.3627529, -2.5794172, 2.6549282
3: -0.8774633, 2.8582191, -0.9510102, 3.1366005, -4.0140638, 3.8092294
4: -1.6534157, 1.4405326, -1.8486843, 1.5311114, -3.1845269, 3.2892170

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
time: 0.38 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.6604972, 1.4221686, -1.8888682, 1.8577981
1: -0.5722542, 1.9478822, -0.7327771, 2.1478353, -2.7200894, 2.6806593
2: -1.2166643, 1.2926129, -1.5855274, 1.5161924, -2.7328568, 2.8781404
3: -0.8774633, 2.8582191, -1.0668936, 3.4070873, -4.2845507, 3.9251127
4: -1.6534157, 1.4405326, -2.0916843, 1.7000986, -3.3535142, 3.5322170

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9209013
time: 0.43 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5382795, 1.2781243, -1.8458304, 1.8629212
1: -0.6357365, 1.9145675, -0.6404734, 2.1168823, -2.7526188, 2.5550408
2: -1.3678999, 1.4330735, -1.3802681, 1.3657906, -2.7336905, 2.8133416
3: -0.9600172, 2.9499054, -0.9617138, 3.1572475, -4.1172647, 3.9116192
4: -1.7817144, 1.5861579, -1.8696117, 1.5350124, -3.3167267, 3.4557695

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.37 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.6682997, 1.4284103, -1.9961164, 1.9929414
1: -0.6357365, 1.9145675, -0.7392297, 2.1551766, -2.7909131, 2.6537971
2: -1.3678999, 1.4330735, -1.5968580, 1.5233715, -2.8912714, 3.0299315
3: -0.9600172, 2.9499054, -1.0718527, 3.4160938, -4.3761110, 4.0217581
4: -1.7817144, 1.5861579, -2.1025352, 1.7087098, -3.4904242, 3.6886930

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.38 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.25 seconds
IS_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9208025, upper bound: 1.9126412
IS_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9208025, upper bound: 1.9147438
IS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9209013
IS_A2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.25
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5306449, 1.2748586, -1.7754699, 1.7758524
1: -0.6059399, 2.0574260, -0.6343837, 2.1120329, -2.7179728, 2.6918097
2: -1.3041267, 1.3373374, -1.3630619, 1.3633302, -2.6674571, 2.7003994
3: -0.9207745, 3.0303936, -0.9513988, 3.1380749, -4.0588493, 3.9817924
4: -1.7729893, 1.4975882, -1.8497353, 1.5318481, -3.3048372, 3.3473234

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.38 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5837932, 1.3274705, -1.8280818, 1.8290007
1: -0.6059399, 2.0574260, -0.6598225, 2.0440698, -2.6500096, 2.7172484
2: -1.3041267, 1.3373374, -1.4269304, 1.4358747, -2.7400014, 2.7642679
3: -0.9207745, 3.0303936, -0.9843640, 3.1207066, -4.0414810, 4.0147576
4: -1.7729893, 1.4975882, -1.8964205, 1.5964544, -3.3694437, 3.3940086

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.38 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5306449, 1.2748586, -1.7571431, 1.7535167
1: -0.5880079, 2.0136833, -0.6343837, 2.1120329, -2.7000408, 2.6480670
2: -1.2619104, 1.3153384, -1.3630619, 1.3633302, -2.6252408, 2.6784003
3: -0.8966851, 2.9580421, -0.9513988, 3.1380749, -4.0347600, 3.9094410
4: -1.7185078, 1.4698515, -1.8497353, 1.5318481, -3.2503557, 3.3195868

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5005965, 1.2284206, -1.7290319, 1.7458041
1: -0.6059399, 2.0574260, -0.6031599, 2.0058041, -2.6117439, 2.6605859
2: -1.3041267, 1.3373374, -1.2835169, 1.3217146, -2.6258414, 2.6208544
3: -0.9207745, 3.0303936, -0.9144611, 2.9694157, -3.8901901, 3.9448547
4: -1.7729893, 1.4975882, -1.7390747, 1.4786390, -3.2516284, 3.2366629

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A2_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166653, upper bound: 1.9064313
time: 0.37 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160093, upper bound: 1.9086312
time: 0.37 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9202833, upper bound: 1.9147438
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9202833, upper bound: 1.9147438
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5218196, 1.2599968, -1.7422813, 1.7446914
1: -0.5880079, 2.0136833, -0.6245918, 2.0851417, -2.6731496, 2.6382751
2: -1.2619104, 1.3153384, -1.3428059, 1.3493736, -2.6112840, 2.6581442
3: -0.8966851, 2.9580421, -0.9403701, 3.0976830, -3.9943681, 3.8984122
4: -1.7185078, 1.4698515, -1.8229065, 1.5146019, -3.2331097, 3.2927580

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5218196, 1.2599968, -1.8102001, 1.8077524
1: -0.6213336, 1.9295921, -0.6245918, 2.0851417, -2.7064753, 2.5541840
2: -1.3427572, 1.4046347, -1.3428059, 1.3493736, -2.6921308, 2.7474406
3: -0.9433498, 2.9237518, -0.9403701, 3.0976830, -4.0410328, 3.8641219
4: -1.7625704, 1.5530781, -1.8229065, 1.5146019, -3.2771723, 3.3759847

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.36 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6604972, 1.4221686, -1.9044532, 1.8833690
1: -0.5880079, 2.0136833, -0.7327771, 2.1478353, -2.7358432, 2.7464604
2: -1.2619104, 1.3153384, -1.5855274, 1.5161924, -2.7781029, 2.9008658
3: -0.8966851, 2.9580421, -1.0668936, 3.4070873, -4.3037724, 4.0249357
4: -1.7185078, 1.4698515, -2.0916843, 1.7000986, -3.4186063, 3.5615358

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.37 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6604972, 1.4221686, -1.9723719, 1.9464300
1: -0.6213336, 1.9295921, -0.7327771, 2.1478353, -2.7691689, 2.6623693
2: -1.3427572, 1.4046347, -1.5855274, 1.5161924, -2.8589497, 2.9901621
3: -0.9433498, 2.9237518, -1.0668936, 3.4070873, -4.3504372, 3.9906454
4: -1.7625704, 1.5530781, -2.0916843, 1.7000986, -3.4626689, 3.6447625

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5302949, 1.2742450, -1.7409446, 1.7275958
1: -0.5722542, 1.9478822, -0.6340547, 2.1110892, -2.6833434, 2.5819368
2: -1.2166643, 1.2926129, -1.3623152, 1.3627529, -2.5794172, 2.6549282
3: -0.8774633, 2.8582191, -0.9510102, 3.1366005, -4.0140638, 3.8092294
4: -1.6534157, 1.4405326, -1.8486843, 1.5311114, -3.1845269, 3.2892170

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9126412, upper bound: 1.9208025
time: 0.35 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5071945, 1.2492044, -0.5302949, 1.2742450, -1.7814395, 1.7794993
1: -0.5821843, 1.8327765, -0.6340547, 2.1110892, -2.6932735, 2.4668312
2: -1.2480936, 1.3721671, -1.3623152, 1.3627529, -2.6108465, 2.7344823
3: -0.9009604, 2.7484131, -0.9510102, 3.1366005, -4.0375609, 3.6994233
4: -1.6378326, 1.5086918, -1.8486843, 1.5311114, -3.1689439, 3.3573761

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9126412, upper bound: 1.9208025
time: 0.34 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.6604972, 1.4221686, -1.8888682, 1.8577981
1: -0.5722542, 1.9478822, -0.7327771, 2.1478353, -2.7200894, 2.6806593
2: -1.2166643, 1.2926129, -1.5855274, 1.5161924, -2.7328568, 2.8781404
3: -0.8774633, 2.8582191, -1.0668936, 3.4070873, -4.2845507, 3.9251127
4: -1.6534157, 1.4405326, -2.0916843, 1.7000986, -3.3535142, 3.5322170

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9147438
time: 0.38 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9147438
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5071945, 1.2492044, -0.6604972, 1.4221686, -1.9293631, 1.9097016
1: -0.5821843, 1.8327765, -0.7327771, 2.1478353, -2.7300196, 2.5655537
2: -1.2480936, 1.3721671, -1.5855274, 1.5161924, -2.7642860, 2.9576945
3: -0.9009604, 2.7484131, -1.0668936, 3.4070873, -4.3080478, 3.8153067
4: -1.6378326, 1.5086918, -2.0916843, 1.7000986, -3.3379312, 3.6003761

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.35 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5382795, 1.2781243, -1.7448239, 1.7355804
1: -0.5722542, 1.9478822, -0.6404734, 2.1168823, -2.6891365, 2.5883555
2: -1.2166643, 1.2926129, -1.3802681, 1.3657906, -2.5824549, 2.6728811
3: -0.8774633, 2.8582191, -0.9617138, 3.1572475, -4.0347109, 3.8199329
4: -1.6534157, 1.4405326, -1.8696117, 1.5350124, -3.1884279, 3.3101444

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9107760, upper bound: 1.9179591
time: 0.35 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099192, upper bound: 1.9167434
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5382795, 1.2781243, -1.8458304, 1.8629212
1: -0.6357365, 1.9145675, -0.6404734, 2.1168823, -2.7526188, 2.5550408
2: -1.3678999, 1.4330735, -1.3802681, 1.3657906, -2.7336905, 2.8133416
3: -0.9600172, 2.9499054, -0.9617138, 3.1572475, -4.1172647, 3.9116192
4: -1.7817144, 1.5861579, -1.8696117, 1.5350124, -3.3167267, 3.4557695

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9208025
time: 0.45 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.6682997, 1.4284103, -1.8951099, 1.8656006
1: -0.5722542, 1.9478822, -0.7392297, 2.1551766, -2.7274308, 2.6871119
2: -1.2166643, 1.2926129, -1.5968580, 1.5233715, -2.7400358, 2.8894711
3: -0.8774633, 2.8582191, -1.0718527, 3.4160938, -4.2935572, 3.9300718
4: -1.6534157, 1.4405326, -2.1025352, 1.7087098, -3.3621254, 3.5430679

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.35 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.6682997, 1.4284103, -1.9961164, 1.9929414
1: -0.6357365, 1.9145675, -0.7392297, 2.1551766, -2.7909131, 2.6537971
2: -1.3678999, 1.4330735, -1.5968580, 1.5233715, -2.8912714, 3.0299315
3: -0.9600172, 2.9499054, -1.0718527, 3.4160938, -4.3761110, 4.0217581
4: -1.7817144, 1.5861579, -2.1025352, 1.7087098, -3.4904242, 3.6886930

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.39 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623
time: 0.48 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.42 seconds
IS_A2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9166653, upper bound: 1.9064313
IS_A2_A1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9160093, upper bound: 1.9086312
IS_A2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9202833, upper bound: 1.9147438
IS_A2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9202833, upper bound: 1.9147438
IS_A2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9228433
IS_A2_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9126412, upper bound: 1.9208025
IS_A2_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9126412, upper bound: 1.9208025
IS_A2_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9147438
IS_A2_A2_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9147438
IS_A2_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9107760, upper bound: 1.9179591
IS_A2_A2_A1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9099192, upper bound: 1.9167434
IS_A2_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9208025
IS_A2_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.42
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5306449, 1.2748586, -1.7754699, 1.7758524
1: -0.6059399, 2.0574260, -0.6343837, 2.1120329, -2.7179728, 2.6918097
2: -1.3041267, 1.3373374, -1.3630619, 1.3633302, -2.6674571, 2.7003994
3: -0.9207745, 3.0303936, -0.9513988, 3.1380749, -4.0588493, 3.9817924
4: -1.7729893, 1.4975882, -1.8497353, 1.5318481, -3.3048372, 3.3473234

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5837932, 1.3274705, -1.8280818, 1.8290007
1: -0.6059399, 2.0574260, -0.6598225, 2.0440698, -2.6500096, 2.7172484
2: -1.3041267, 1.3373374, -1.4269304, 1.4358747, -2.7400014, 2.7642679
3: -0.9207745, 3.0303936, -0.9843640, 3.1207066, -4.0414810, 4.0147576
4: -1.7729893, 1.4975882, -1.8964205, 1.5964544, -3.3694437, 3.3940086

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5193291, 1.2552297, -1.7558410, 1.7645366
1: -0.6059399, 2.0574260, -0.6218777, 2.0725369, -2.6784768, 2.6793036
2: -1.3041267, 1.3373374, -1.3364878, 1.3432146, -2.6473413, 2.6738253
3: -0.9207745, 3.0303936, -0.9367762, 3.0830059, -4.0037804, 3.9671698
4: -1.7729893, 1.4975882, -1.8136654, 1.5062159, -3.2792053, 3.3112535

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5837932, 1.3274705, -1.8280818, 1.8290007
1: -0.6059399, 2.0574260, -0.6598225, 2.0440698, -2.6500096, 2.7172484
2: -1.3041267, 1.3373374, -1.4269304, 1.4358747, -2.7400014, 2.7642679
3: -0.9207745, 3.0303936, -0.9843640, 3.1207066, -4.0414810, 4.0147576
4: -1.7729893, 1.4975882, -1.8964205, 1.5964544, -3.3694437, 3.3940086

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5193291, 1.2552297, -1.8054330, 1.8052619
1: -0.6213336, 1.9295921, -0.6218777, 2.0725369, -2.6938705, 2.5514698
2: -1.3427572, 1.4046347, -1.3364878, 1.3432146, -2.6859717, 2.7411225
3: -0.9433498, 2.9237518, -0.9367762, 3.0830059, -4.0263557, 3.8605280
4: -1.7625704, 1.5530781, -1.8136654, 1.5062159, -3.2687864, 3.3667436

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5306449, 1.2748586, -1.7571431, 1.7535167
1: -0.5880079, 2.0136833, -0.6343837, 2.1120329, -2.7000408, 2.6480670
2: -1.2619104, 1.3153384, -1.3630619, 1.3633302, -2.6252408, 2.6784003
3: -0.8966851, 2.9580421, -0.9513988, 3.1380749, -4.0347600, 3.9094410
4: -1.7185078, 1.4698515, -1.8497353, 1.5318481, -3.2503557, 3.3195868

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5193291, 1.2552297, -1.7375143, 1.7422009
1: -0.5880079, 2.0136833, -0.6218777, 2.0725369, -2.6605449, 2.6355610
2: -1.2619104, 1.3153384, -1.3364878, 1.3432146, -2.6051250, 2.6518261
3: -0.8966851, 2.9580421, -0.9367762, 3.0830059, -3.9796910, 3.8948183
4: -1.7185078, 1.4698515, -1.8136654, 1.5062159, -3.2247238, 3.2835169

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5193291, 1.2552297, -1.8054330, 1.8052619
1: -0.6213336, 1.9295921, -0.6218777, 2.0725369, -2.6938705, 2.5514698
2: -1.3427572, 1.4046347, -1.3364878, 1.3432146, -2.6859717, 2.7411225
3: -0.9433498, 2.9237518, -0.9367762, 3.0830059, -4.0263557, 3.8605280
4: -1.7625704, 1.5530781, -1.8136654, 1.5062159, -3.2687864, 3.3667436

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5999899, 1.3601222, -1.8607335, 1.8451974
1: -0.6059399, 2.0574260, -0.6711960, 2.0180054, -2.6239452, 2.7286220
2: -1.3041267, 1.3373374, -1.4441347, 1.4615185, -2.7656453, 2.7814722
3: -0.9207745, 3.0303936, -0.9964314, 3.1247883, -4.0455627, 4.0268250
4: -1.7729893, 1.4975882, -1.9009428, 1.6262220, -3.3992114, 3.3985310

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5999899, 1.3601222, -1.9103255, 1.8859227
1: -0.6213336, 1.9295921, -0.6711960, 2.0180054, -2.6393390, 2.6007881
2: -1.3427572, 1.4046347, -1.4441347, 1.4615185, -2.8042758, 2.8487694
3: -0.9433498, 2.9237518, -0.9964314, 3.1247883, -4.0681381, 3.9201832
4: -1.7625704, 1.5530781, -1.9009428, 1.6262220, -3.3887925, 3.4540210

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9115793
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5218196, 1.2599968, -1.7422813, 1.7446914
1: -0.5880079, 2.0136833, -0.6245918, 2.0851417, -2.6731496, 2.6382751
2: -1.2619104, 1.3153384, -1.3428059, 1.3493736, -2.6112840, 2.6581442
3: -0.8966851, 2.9580421, -0.9403701, 3.0976830, -3.9943681, 3.8984122
4: -1.7185078, 1.4698515, -1.8229065, 1.5146019, -3.2331097, 3.2927580

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5218196, 1.2599968, -1.8102001, 1.8077524
1: -0.6213336, 1.9295921, -0.6245918, 2.0851417, -2.7064753, 2.5541840
2: -1.3427572, 1.4046347, -1.3428059, 1.3493736, -2.6921308, 2.7474406
3: -0.9433498, 2.9237518, -0.9403701, 3.0976830, -4.0410328, 3.8641219
4: -1.7625704, 1.5530781, -1.8229065, 1.5146019, -3.2771723, 3.3759847

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5218196, 1.2599968, -1.7422813, 1.7446914
1: -0.5880079, 2.0136833, -0.6245918, 2.0851417, -2.6731496, 2.6382751
2: -1.2619104, 1.3153384, -1.3428059, 1.3493736, -2.6112840, 2.6581442
3: -0.8966851, 2.9580421, -0.9403701, 3.0976830, -3.9943681, 3.8984122
4: -1.7185078, 1.4698515, -1.8229065, 1.5146019, -3.2331097, 3.2927580

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.6604972, 1.4221686, -1.9044532, 1.8833690
1: -0.5880079, 2.0136833, -0.7327771, 2.1478353, -2.7358432, 2.7464604
2: -1.2619104, 1.3153384, -1.5855274, 1.5161924, -2.7781029, 2.9008658
3: -0.8966851, 2.9580421, -1.0668936, 3.4070873, -4.3037724, 4.0249357
4: -1.7185078, 1.4698515, -2.0916843, 1.7000986, -3.4186063, 3.5615358

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5218196, 1.2599968, -1.8102001, 1.8077524
1: -0.6213336, 1.9295921, -0.6245918, 2.0851417, -2.7064753, 2.5541840
2: -1.3427572, 1.4046347, -1.3428059, 1.3493736, -2.6921308, 2.7474406
3: -0.9433498, 2.9237518, -0.9403701, 3.0976830, -4.0410328, 3.8641219
4: -1.7625704, 1.5530781, -1.8229065, 1.5146019, -3.2771723, 3.3759847

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.6604972, 1.4221686, -1.9723719, 1.9464300
1: -0.6213336, 1.9295921, -0.7327771, 2.1478353, -2.7691689, 2.6623693
2: -1.3427572, 1.4046347, -1.5855274, 1.5161924, -2.8589497, 2.9901621
3: -0.9433498, 2.9237518, -1.0668936, 3.4070873, -4.3504372, 3.9906454
4: -1.7625704, 1.5530781, -2.0916843, 1.7000986, -3.4626689, 3.6447625

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5302949, 1.2742450, -1.7409446, 1.7275958
1: -0.5722542, 1.9478822, -0.6340547, 2.1110892, -2.6833434, 2.5819368
2: -1.2166643, 1.2926129, -1.3623152, 1.3627529, -2.5794172, 2.6549282
3: -0.8774633, 2.8582191, -0.9510102, 3.1366005, -4.0140638, 3.8092294
4: -1.6534157, 1.4405326, -1.8486843, 1.5311114, -3.1845269, 3.2892170

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
time: 0.44 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5837932, 1.3274705, -1.7941701, 1.7810941
1: -0.5722542, 1.9478822, -0.6598225, 2.0440698, -2.6163239, 2.6077046
2: -1.2166643, 1.2926129, -1.4269304, 1.4358747, -2.6525390, 2.7195435
3: -0.8774633, 2.8582191, -0.9843640, 3.1207066, -3.9981699, 3.8425832
4: -1.6534157, 1.4405326, -1.8964205, 1.5964544, -3.2498701, 3.3369532

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
time: 0.42 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5071945, 1.2492044, -0.5302949, 1.2742450, -1.7814395, 1.7794993
1: -0.5821843, 1.8327765, -0.6340547, 2.1110892, -2.6932735, 2.4668312
2: -1.2480936, 1.3721671, -1.3623152, 1.3627529, -2.6108465, 2.7344823
3: -0.9009604, 2.7484131, -0.9510102, 3.1366005, -4.0375609, 3.6994233
4: -1.6378326, 1.5086918, -1.8486843, 1.5311114, -3.1689439, 3.3573761

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5071945, 1.2492044, -0.5837932, 1.3274705, -1.8346651, 1.8329976
1: -0.5821843, 1.8327765, -0.6598225, 2.0440698, -2.6262541, 2.4925990
2: -1.2480936, 1.3721671, -1.4269304, 1.4358747, -2.6839683, 2.7990975
3: -0.9009604, 2.7484131, -0.9843640, 3.1207066, -4.0216670, 3.7327771
4: -1.6378326, 1.5086918, -1.8964205, 1.5964544, -3.2342870, 3.4051123

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.39 seconds

## Relational analysis of IS_A2_A2_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5071945, 1.2492044, -0.5218196, 1.2599968, -1.7671913, 1.7710240
1: -0.5821843, 1.8327765, -0.6245918, 2.0851417, -2.6673260, 2.4573684
2: -1.2480936, 1.3721671, -1.3428059, 1.3493736, -2.5974672, 2.7149730
3: -0.9009604, 2.7484131, -0.9403701, 3.0976830, -3.9986434, 3.6887832
4: -1.6378326, 1.5086918, -1.8229065, 1.5146019, -3.1524346, 3.3315983

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.39 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5071945, 1.2492044, -0.6604972, 1.4221686, -1.9293631, 1.9097016
1: -0.5821843, 1.8327765, -0.7327771, 2.1478353, -2.7300196, 2.5655537
2: -1.2480936, 1.3721671, -1.5855274, 1.5161924, -2.7642860, 2.9576945
3: -0.9009604, 2.7484131, -1.0668936, 3.4070873, -4.3080478, 3.8153067
4: -1.6378326, 1.5086918, -2.0916843, 1.7000986, -3.3379312, 3.6003761

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.39 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.3880787, 1.1199733, -0.5382795, 1.2781243, -1.6662030, 1.6582528
1: -0.4901099, 1.8158956, -0.6404734, 2.1168823, -2.6069922, 2.4563689
2: -1.0379610, 1.2225698, -1.3802681, 1.3657906, -2.4037516, 2.6028380
3: -0.7823596, 2.5484247, -0.9617138, 3.1572475, -3.9396071, 3.5101385
4: -1.4211206, 1.3328046, -1.8696117, 1.5350124, -2.9561329, 3.2024164

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9025367, upper bound: 1.9170282
time: 0.38 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9067351, upper bound: 1.9160092
time: 0.40 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9067351, upper bound: 1.9160092
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5382795, 1.2781243, -1.8458304, 1.8629212
1: -0.6357365, 1.9145675, -0.6404734, 2.1168823, -2.7526188, 2.5550408
2: -1.3678999, 1.4330735, -1.3802681, 1.3657906, -2.7336905, 2.8133416
3: -0.9600172, 2.9499054, -0.9617138, 3.1572475, -4.1172647, 3.9116192
4: -1.7817144, 1.5861579, -1.8696117, 1.5350124, -3.3167267, 3.4557695

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5837932, 1.3274705, -1.8951766, 1.9084349
1: -0.6357365, 1.9145675, -0.6598225, 2.0440698, -2.6798062, 2.5743899
2: -1.3678999, 1.4330735, -1.4269304, 1.4358747, -2.8037746, 2.8600039
3: -0.9600172, 2.9499054, -0.9843640, 3.1207066, -4.0807238, 3.9342694
4: -1.7817144, 1.5861579, -1.8964205, 1.5964544, -3.3781688, 3.4825783

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.41 seconds

## Relational analysis of IS_A2_A2_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.5218196, 1.2599968, -1.7266964, 1.7191205
1: -0.5722542, 1.9478822, -0.6245918, 2.0851417, -2.6573958, 2.5724740
2: -1.2166643, 1.2926129, -1.3428059, 1.3493736, -2.5660379, 2.6354189
3: -0.8774633, 2.8582191, -0.9403701, 3.0976830, -3.9751463, 3.7985892
4: -1.6534157, 1.4405326, -1.8229065, 1.5146019, -3.1680176, 3.2634392

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
time: 0.42 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4666996, 1.1973009, -0.6682997, 1.4284103, -1.8951099, 1.8656006
1: -0.5722542, 1.9478822, -0.7392297, 2.1551766, -2.7274308, 2.6871119
2: -1.2166643, 1.2926129, -1.5968580, 1.5233715, -2.7400358, 2.8894711
3: -0.8774633, 2.8582191, -1.0718527, 3.4160938, -4.2935572, 3.9300718
4: -1.6534157, 1.4405326, -2.1025352, 1.7087098, -3.3621254, 3.5430679

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9209013
time: 0.44 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.5218196, 1.2599968, -1.8277029, 1.8464613
1: -0.6357365, 1.9145675, -0.6245918, 2.0851417, -2.7208781, 2.5391593
2: -1.3678999, 1.4330735, -1.3428059, 1.3493736, -2.7172735, 2.7758794
3: -0.9600172, 2.9499054, -0.9403701, 3.0976830, -4.0577002, 3.8902755
4: -1.7817144, 1.5861579, -1.8229065, 1.5146019, -3.2963164, 3.4090643

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
time: 0.40 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5677061, 1.3246417, -0.6682997, 1.4284103, -1.9961164, 1.9929414
1: -0.6357365, 1.9145675, -0.7392297, 2.1551766, -2.7909131, 2.6537971
2: -1.3678999, 1.4330735, -1.5968580, 1.5233715, -2.8912714, 3.0299315
3: -0.9600172, 2.9499054, -1.0718527, 3.4160938, -4.3761110, 4.0217581
4: -1.7817144, 1.5861579, -2.1025352, 1.7087098, -3.4904242, 3.6886930

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
time: 0.39 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623
time: 0.42 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.50 seconds
IS_A2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9115793
IS_A2_A1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
IS_A2_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9115793
IS_A2_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
IS_A2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A2_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9067351, upper bound: 1.9160092
IS_A2_A2_A1_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9067351, upper bound: 1.9160092
IS_A2_A2_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5306449, 1.2748586, -1.7754699, 1.7758524
1: -0.6059399, 2.0574260, -0.6343837, 2.1120329, -2.7179728, 2.6918097
2: -1.3041267, 1.3373374, -1.3630619, 1.3633302, -2.6674571, 2.7003994
3: -0.9207745, 3.0303936, -0.9513988, 3.1380749, -4.0588493, 3.9817924
4: -1.7729893, 1.4975882, -1.8497353, 1.5318481, -3.3048372, 3.3473234

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5837932, 1.3274705, -1.8280818, 1.8290007
1: -0.6059399, 2.0574260, -0.6598225, 2.0440698, -2.6500096, 2.7172484
2: -1.3041267, 1.3373374, -1.4269304, 1.4358747, -2.7400014, 2.7642679
3: -0.9207745, 3.0303936, -0.9843640, 3.1207066, -4.0414810, 4.0147576
4: -1.7729893, 1.4975882, -1.8964205, 1.5964544, -3.3694437, 3.3940086

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5306449, 1.2748586, -1.7571431, 1.7535167
1: -0.5880079, 2.0136833, -0.6343837, 2.1120329, -2.7000408, 2.6480670
2: -1.2619104, 1.3153384, -1.3630619, 1.3633302, -2.6252408, 2.6784003
3: -0.8966851, 2.9580421, -0.9513988, 3.1380749, -4.0347600, 3.9094410
4: -1.7185078, 1.4698515, -1.8497353, 1.5318481, -3.2503557, 3.3195868

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.38 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5193291, 1.2552297, -1.7558410, 1.7645366
1: -0.6059399, 2.0574260, -0.6218777, 2.0725369, -2.6784768, 2.6793036
2: -1.3041267, 1.3373374, -1.3364878, 1.3432146, -2.6473413, 2.6738253
3: -0.9207745, 3.0303936, -0.9367762, 3.0830059, -4.0037804, 3.9671698
4: -1.7729893, 1.4975882, -1.8136654, 1.5062159, -3.2792053, 3.3112535

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5193291, 1.2552297, -1.8054330, 1.8052619
1: -0.6213336, 1.9295921, -0.6218777, 2.0725369, -2.6938705, 2.5514698
2: -1.3427572, 1.4046347, -1.3364878, 1.3432146, -2.6859717, 2.7411225
3: -0.9433498, 2.9237518, -0.9367762, 3.0830059, -4.0263557, 3.8605280
4: -1.7625704, 1.5530781, -1.8136654, 1.5062159, -3.2687864, 3.3667436

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5006113, 1.2452075, -0.5837932, 1.3274705, -1.8280818, 1.8290007
1: -0.6059399, 2.0574260, -0.6598225, 2.0440698, -2.6500096, 2.7172484
2: -1.3041267, 1.3373374, -1.4269304, 1.4358747, -2.7400014, 2.7642679
3: -0.9207745, 3.0303936, -0.9843640, 3.1207066, -4.0414810, 4.0147576
4: -1.7729893, 1.4975882, -1.8964205, 1.5964544, -3.3694437, 3.3940086

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5193291, 1.2552297, -1.7375143, 1.7422009
1: -0.5880079, 2.0136833, -0.6218777, 2.0725369, -2.6605449, 2.6355610
2: -1.2619104, 1.3153384, -1.3364878, 1.3432146, -2.6051250, 2.6518261
3: -0.8966851, 2.9580421, -0.9367762, 3.0830059, -3.9796910, 3.8948183
4: -1.7185078, 1.4698515, -1.8136654, 1.5062159, -3.2247238, 3.2835169

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5193291, 1.2552297, -1.8054330, 1.8052619
1: -0.6213336, 1.9295921, -0.6218777, 2.0725369, -2.6938705, 2.5514698
2: -1.3427572, 1.4046347, -1.3364878, 1.3432146, -2.6859717, 2.7411225
3: -0.9433498, 2.9237518, -0.9367762, 3.0830059, -4.0263557, 3.8605280
4: -1.7625704, 1.5530781, -1.8136654, 1.5062159, -3.2687864, 3.3667436

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5306449, 1.2748586, -1.7571431, 1.7535167
1: -0.5880079, 2.0136833, -0.6343837, 2.1120329, -2.7000408, 2.6480670
2: -1.2619104, 1.3153384, -1.3630619, 1.3633302, -2.6252408, 2.6784003
3: -0.8966851, 2.9580421, -0.9513988, 3.1380749, -4.0347600, 3.9094410
4: -1.7185078, 1.4698515, -1.8497353, 1.5318481, -3.2503557, 3.3195868

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5306449, 1.2748586, -1.8031417, 1.8114083
1: -0.6040082, 1.9247079, -0.6343837, 2.1120329, -2.7160411, 2.5590916
2: -1.3036880, 1.4001466, -1.3630619, 1.3633302, -2.6670184, 2.7632084
3: -0.9276228, 2.8863373, -0.9513988, 3.1380749, -4.0656977, 3.8377361
4: -1.7242384, 1.5467284, -1.8497353, 1.5318481, -3.2560863, 3.3964636

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.5282831, 1.2807634, -0.5837932, 1.3274705, -1.8557537, 1.8645566
1: -0.6040082, 1.9247079, -0.6598225, 2.0440698, -2.6480780, 2.5845304
2: -1.3036880, 1.4001466, -1.4269304, 1.4358747, -2.7395627, 2.8270769
3: -0.9276228, 2.8863373, -0.9843640, 3.1207066, -4.0483294, 3.8707013
4: -1.7242384, 1.5467284, -1.8964205, 1.5964544, -3.3206928, 3.4431489

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.38 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5306449, 1.2748586, -1.7571431, 1.7535167
1: -0.5880079, 2.0136833, -0.6343837, 2.1120329, -2.7000408, 2.6480670
2: -1.2619104, 1.3153384, -1.3630619, 1.3633302, -2.6252408, 2.6784003
3: -0.8966851, 2.9580421, -0.9513988, 3.1380749, -4.0347600, 3.9094410
4: -1.7185078, 1.4698515, -1.8497353, 1.5318481, -3.2503557, 3.3195868

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5306449, 1.2748586, -1.8250619, 1.8165777
1: -0.6213336, 1.9295921, -0.6343837, 2.1120329, -2.7333665, 2.5639758
2: -1.3427572, 1.4046347, -1.3630619, 1.3633302, -2.7060876, 2.7676966
3: -0.9433498, 2.9237518, -0.9513988, 3.1380749, -4.0814247, 3.8751507
4: -1.7625704, 1.5530781, -1.8497353, 1.5318481, -3.2944183, 3.4028134

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5837932, 1.3274705, -1.8097551, 1.8066649
1: -0.5880079, 2.0136833, -0.6598225, 2.0440698, -2.6320777, 2.6735058
2: -1.2619104, 1.3153384, -1.4269304, 1.4358747, -2.6977851, 2.7422688
3: -0.8966851, 2.9580421, -0.9843640, 3.1207066, -4.0173917, 3.9424062
4: -1.7185078, 1.4698515, -1.8964205, 1.5964544, -3.3149621, 3.3662720

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5502033, 1.2859328, -0.5837932, 1.3274705, -1.8776739, 1.8697259
1: -0.6213336, 1.9295921, -0.6598225, 2.0440698, -2.6654034, 2.5894146
2: -1.3427572, 1.4046347, -1.4269304, 1.4358747, -2.7786319, 2.8315651
3: -0.9433498, 2.9237518, -0.9843640, 3.1207066, -4.0640564, 3.9081159
4: -1.7625704, 1.5530781, -1.8964205, 1.5964544, -3.3590248, 3.4494987

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4822845, 1.2228718, -0.5193291, 1.2552297, -1.7375143, 1.7422009
1: -0.5880079, 2.0136833, -0.6218777, 2.0725369, -2.6605449, 2.6355610
2: -1.2619104, 1.3153384, -1.3364878, 1.3432146, -2.6051250, 2.6518261
3: -0.8966851, 2.9580421, -0.9367762, 3.0830059, -3.9796910, 3.8948183
4: -1.7185078, 1.4698515, -1.8136654, 1.5062159, -3.2247238, 3.2835169

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
time: 0.41 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.98 seconds
IS_A2_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9214925
IS_A2_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9115793
IS_A2_A1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
IS_A2_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9115793
IS_A2_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9198879, upper bound: 1.9147438
IS_A2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9214925, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9207285, upper bound: 1.9207285
IS_A2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9210894, upper bound: 1.9228433
IS_A2_A2_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9118746, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9198879
IS_A2_A2_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9202833
IS_A2_A2_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9115795, upper bound: 1.9209013
IS_A2_A2_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.98
Output dim: 0, lower bound: -1.9147439, upper bound: 1.9228623
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=2.269134044647217
rel_dist={0: [-1.927168397752424, 1.9271683977524248]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1153.95 seconds
