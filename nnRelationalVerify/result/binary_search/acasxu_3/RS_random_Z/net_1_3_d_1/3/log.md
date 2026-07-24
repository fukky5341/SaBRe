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
execution time: IAR + LP analysis = 1.42 + 1.10 = 2.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.9315851, upper bound: 1.9315851


# Binary Search by BASE starts (time budget: 1197.47 seconds, max iter: 100)

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
Binary search time: 43.39 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1154.08 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9121549
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.63
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9121549
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9170718, upper bound: 1.9179606
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9192210, upper bound: 1.9145079
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.01 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9170718, upper bound: 1.9179606
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9192210, upper bound: 1.9145079
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9172919
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9080044, upper bound: 1.9066018
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9097187, upper bound: 1.9099032
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9180951, upper bound: 1.9140879
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9145044
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9080044, upper bound: 1.9066018
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9112284, upper bound: 1.9098848
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.46 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9172919
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9080044, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9097187, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9180951, upper bound: 1.9140879
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9145044
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9080044, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -1.9112284, upper bound: 1.9098848

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9176358, upper bound: 1.9106786
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9058248
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9079574, upper bound: 1.9080810
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9105563, upper bound: 1.9050081
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129553, upper bound: 1.9065706
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.13 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9176358, upper bound: 1.9106786
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9058248
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9079574, upper bound: 1.9080810
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9105563, upper bound: 1.9050081
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9129553, upper bound: 1.9065706

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9127665, upper bound: 1.9065706
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9070909
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9070910
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.03 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.9127665, upper bound: 1.9065706
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9070909
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.03
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9070910
Binary search (step 0): status=Status.VERIFIED, low=0.0909091, high=0.1818182, mid=0.0909091, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 1) starts
Candidate diff: 0.1363636


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161542, upper bound: 1.9160467
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9183790
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9161542, upper bound: 1.9160467
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9183790
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9183790
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9181879
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9071747
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9069739
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9183790
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9181879
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9071747
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.49
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9069739

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9129553
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9181879
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9180872
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9129553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9181879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9180872

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9096645
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065706, upper bound: 1.9127665
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9076650
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9096645
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -1.9065706, upper bound: 1.9127665
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9076650
Binary search (step 1): status=Status.VERIFIED, low=0.1363636, high=0.1818182, mid=0.1363636, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 2) starts
Candidate diff: 0.1590909


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
time: 0.41 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285226, upper bound: 1.9314710
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9189160
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9285226, upper bound: 1.9314710
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9189160

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9110647, upper bound: 1.9195167
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9192868
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171793, upper bound: 1.9164153
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171844, upper bound: 1.9163851
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178415, upper bound: 1.9271448
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9282813, upper bound: 1.9295301
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9187070
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187070, upper bound: 1.9188477
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9110647, upper bound: 1.9195167
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9192868
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9171793, upper bound: 1.9164153
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9171844, upper bound: 1.9163851
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9178415, upper bound: 1.9271448
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9282813, upper bound: 1.9295301
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9187070
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9187070, upper bound: 1.9188477

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9160467
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9183790
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9192868
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9145079, upper bound: 1.9192210
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9164153
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171793, upper bound: 1.9162936
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9126185
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9086313
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9079029, upper bound: 1.9064814
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9079029, upper bound: 1.9051840
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9278691, upper bound: 1.9216869
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9222674, upper bound: 1.9285013
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9176358, upper bound: 1.9106786
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183790, upper bound: 1.9106786
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9212147, upper bound: 1.9188477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227115, upper bound: 1.9181005
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9160467
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9183790
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9192868
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9145079, upper bound: 1.9192210
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9164153
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9171793, upper bound: 1.9162936
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9126185
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9086313
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9079029, upper bound: 1.9064814
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9079029, upper bound: 1.9051840
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9278691, upper bound: 1.9216869
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9222674, upper bound: 1.9285013
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9176358, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9183790, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9212147, upper bound: 1.9188477
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -1.9227115, upper bound: 1.9181005

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9129553
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9151243, upper bound: 1.9154909
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9122663
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070585, upper bound: 1.9150024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9051840, upper bound: 1.9079029
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9071747, upper bound: 1.9126043
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9034557
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9219812, upper bound: 1.9216787
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9278691, upper bound: 1.9183393
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9285013
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9222674, upper bound: 1.9233079
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9176358, upper bound: 1.9106786
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9168757, upper bound: 1.9106786
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9165597, upper bound: 1.9076207
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9106786
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9106786
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113211, upper bound: 1.9108930
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113211, upper bound: 1.9108930
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9129553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9151243, upper bound: 1.9154909
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9122663
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9070585, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9051840, upper bound: 1.9079029
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9071747, upper bound: 1.9126043
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9034557
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9219812, upper bound: 1.9216787
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9278691, upper bound: 1.9183393
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9285013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9222674, upper bound: 1.9233079
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9176358, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9168757, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9165597, upper bound: 1.9076207
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9113211, upper bound: 1.9108930
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9113211, upper bound: 1.9108930

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9248490, upper bound: 1.9181650
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9248490, upper bound: 1.9166054
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9118065, upper bound: 1.9108930
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9076755, upper bound: 1.9080810
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9070543
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9212821
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9186595, upper bound: 1.9221345
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9070909
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9070910
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.65 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9248490, upper bound: 1.9181650
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9248490, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9118065, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9076755, upper bound: 1.9080810
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9070543
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9212821
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9186595, upper bound: 1.9221345
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9070909
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.65
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9070910

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108040, upper bound: 1.9050319
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108040, upper bound: 1.9050319
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113568, upper bound: 1.9034504
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9066648
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9036108, upper bound: 1.9065718
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9058468, upper bound: 1.9068932
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9058468, upper bound: 1.9068642
time: 0.39 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.28 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9108040, upper bound: 1.9050319
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9108040, upper bound: 1.9050319
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9113568, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9066648
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9036108, upper bound: 1.9065718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9058468, upper bound: 1.9068932
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.28
Output dim: 0, lower bound: -1.9058468, upper bound: 1.9068642
Binary search (step 2): status=Status.VERIFIED, low=0.1590909, high=0.1818182, mid=0.1590909, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 3) starts
Candidate diff: 0.1704546


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9287598
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9285347
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9287598
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9285347
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9280522
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9280543
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171152, upper bound: 1.9285347
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9291097, upper bound: 1.9250327
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178415, upper bound: 1.9271448
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9187064
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9280522
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9171152, upper bound: 1.9285347
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9291097, upper bound: 1.9250327
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9178415, upper bound: 1.9271448
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9187064
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9126043
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9126043
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9083481
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9082598
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280522
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9236749
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9291097, upper bound: 1.9250327
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120257
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121536
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9108930
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9188614, upper bound: 1.9270107
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9126043
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9126043
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9083481
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9082598
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280522
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9236749
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9291097, upper bound: 1.9250327
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120257
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121536
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -1.9188614, upper bound: 1.9270107

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9125640
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9123495
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9088845
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9086313
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9234938
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9167719
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9249909
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218332, upper bound: 1.9167719
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9188544, upper bound: 1.9166054
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9186646, upper bound: 1.9270107
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177468, upper bound: 1.9166411
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9125640
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9123495
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9088845
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9086313
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9234938
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9167719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9249909
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9218332, upper bound: 1.9167719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9188544, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9186646, upper bound: 1.9270107
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9177468, upper bound: 1.9166411

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9063572
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9037431
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9270107, upper bound: 1.9169861
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181650, upper bound: 1.9248490
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9047733
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9047733
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9237651, upper bound: 1.9166054
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187708, upper bound: 1.9166054
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9058468, upper bound: 1.9068932
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068642
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068470, upper bound: 1.9044216
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9044216
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9063572
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9037431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9270107, upper bound: 1.9169861
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9181650, upper bound: 1.9248490
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9047733
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9047733
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9237651, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9187708, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9058468, upper bound: 1.9068932
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9068470, upper bound: 1.9044216
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.22
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9044216

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9054490, upper bound: 1.9034504
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9036096
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072411, upper bound: 1.9034504
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082624, upper bound: 1.9034504
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9087141, upper bound: 1.9034504
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9054490, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9036096
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9072411, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9082624, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 0, lower bound: -1.9087141, upper bound: 1.9034504
Binary search (step 3): status=Status.VERIFIED, low=0.1704546, high=0.1818182, mid=0.1704546, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 4) starts
Candidate diff: 0.1761364


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121541
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9280543
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218332, upper bound: 1.9280543
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178415, upper bound: 1.9271448
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9187064
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121541
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9218332, upper bound: 1.9280543
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9178415, upper bound: 1.9271448
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9187064
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.19
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9168600, upper bound: 1.9280543
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9167719
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218332, upper bound: 1.9233314
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177982, upper bound: 1.9216805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9265637
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9108930
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9168600, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9167719
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9218332, upper bound: 1.9233314
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9177982, upper bound: 1.9216805
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9265637
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.13
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9039643, upper bound: 1.9083481
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9082598
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9034557
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9167719
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9088845
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9086313
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9216773
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9177302
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9265637
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9189582
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.02 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9039643, upper bound: 1.9083481
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9082598
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9167719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9088845
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9086313
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9216773
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9177302
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9265637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9189582

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9085366
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177468
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120257
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121536
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9172739
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 1.99 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9085366
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177468
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9050081
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120257
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9172739

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9040293
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 1.97 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 1.97
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 1.97
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 1.97
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9040293
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 1.97
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
Binary search (step 4): status=Status.VERIFIED, low=0.1761364, high=0.1818182, mid=0.1761364, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 5) starts
Candidate diff: 0.1789773


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121549
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9226944
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9300951
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.94 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 1.94
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 1.94
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121549
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.94
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9226944
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.94
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9300951

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113228, upper bound: 1.9113205
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113228, upper bound: 1.9113205
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9173206
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160467, upper bound: 1.9161542
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 1.96 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 1.96
Output dim: 0, lower bound: -1.9113228, upper bound: 1.9113205
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 1.96
Output dim: 0, lower bound: -1.9113228, upper bound: 1.9113205
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9173206
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 1.96
Output dim: 0, lower bound: -1.9160467, upper bound: 1.9161542

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9172919
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9171918
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.31 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9172919
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.31
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9171918

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068904, upper bound: 1.9058248
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9080810
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072462, upper bound: 1.9063323
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9084936, upper bound: 1.9090062
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 1.95 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 1.95
Output dim: 0, lower bound: -1.9068904, upper bound: 1.9058248
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 1.95
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9080810
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 1.95
Output dim: 0, lower bound: -1.9072462, upper bound: 1.9063323
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 1.95
Output dim: 0, lower bound: -1.9084936, upper bound: 1.9090062
Binary search (step 5): status=Status.VERIFIED, low=0.1789773, high=0.1818182, mid=0.1789773, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 6) starts
Candidate diff: 0.1803977


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9309287
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9273981
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9291097
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9290107
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9309287
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9273981
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9291097
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9290107

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179606, upper bound: 1.9170718
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9300951, upper bound: 1.9221858
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9273981
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9250327, upper bound: 1.9291097
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9171152
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9251325, upper bound: 1.9290107
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9171152
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9179606, upper bound: 1.9170718
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9300951, upper bound: 1.9221858
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9273981
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9250327, upper bound: 1.9291097
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9171152
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9251325, upper bound: 1.9290107
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9171152

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161542, upper bound: 1.9160467
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9183790
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166692, upper bound: 1.9149324
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9165087, upper bound: 1.9070926
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9188477, upper bound: 1.9212147
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226475, upper bound: 1.9221845
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179098, upper bound: 1.9268881
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226475, upper bound: 1.9273981
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161935, upper bound: 1.9171844
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161935, upper bound: 1.9150670
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280383, upper bound: 1.9169282
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172577, upper bound: 1.9171084
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9218290
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9249909, upper bound: 1.9276301
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280588, upper bound: 1.9169116
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190406, upper bound: 1.9171084
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9161542, upper bound: 1.9160467
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9183790
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9166692, upper bound: 1.9149324
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9165087, upper bound: 1.9070926
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9188477, upper bound: 1.9212147
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9226475, upper bound: 1.9221845
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9179098, upper bound: 1.9268881
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9226475, upper bound: 1.9273981
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9161935, upper bound: 1.9171844
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9161935, upper bound: 1.9150670
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9280383, upper bound: 1.9169282
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9172577, upper bound: 1.9171084
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9218290
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9249909, upper bound: 1.9276301
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9280588, upper bound: 1.9169116
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9190406, upper bound: 1.9171084

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9103646
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9094921
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9106786
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9155832
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9219812
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9265637, upper bound: 1.9177302
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9268152
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9241312
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9118061
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9118065
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9156593, upper bound: 1.9122361
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9096788, upper bound: 1.9161233
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9070909
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172577, upper bound: 1.9171084
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9169116, upper bound: 1.9169116
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9040293
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9071747
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9069739
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9132347, upper bound: 1.9077336
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9132347, upper bound: 1.9079385
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9103646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9094921
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9106786
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9155832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9219812
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9265637, upper bound: 1.9177302
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9268152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9241312
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9118061
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9118065
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9156593, upper bound: 1.9122361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9096788, upper bound: 1.9161233
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9070909
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9172577, upper bound: 1.9171084
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9169116, upper bound: 1.9169116
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9040293
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9071747
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9069739
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9132347, upper bound: 1.9077336
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -1.9132347, upper bound: 1.9079385

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9186447
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9270107, upper bound: 1.9169861
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217213, upper bound: 1.9166054
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217465, upper bound: 1.9166054
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9228392
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166371, upper bound: 1.9233292
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098363, upper bound: 1.9070909
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070910
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9166054
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183119, upper bound: 1.9166054
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9256487, upper bound: 1.9166054
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177468, upper bound: 1.9166054
time: 0.39 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9186447
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9270107, upper bound: 1.9169861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9217213, upper bound: 1.9166054
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9217465, upper bound: 1.9166054
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9228392
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9166371, upper bound: 1.9233292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9098363, upper bound: 1.9070909
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070910
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9183119, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9256487, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.13
Output dim: 0, lower bound: -1.9177468, upper bound: 1.9166054

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9058314, upper bound: 1.9057743
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9054490, upper bound: 1.9034504
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9036096
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9039532, upper bound: 1.9034504
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9083410
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082598
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9085061
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9084923
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9034504
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9083410, upper bound: 1.9034504
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115780, upper bound: 1.9034504
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.42 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9058314, upper bound: 1.9057743
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9054490, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9036096
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9039532, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9083410
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082598
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9085061
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9084923
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9083410, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9115780, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.36
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
Binary search (step 6): status=Status.VERIFIED, low=0.1803977, high=0.1818182, mid=0.1803977, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 7) starts
Candidate diff: 0.1811080


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9227628
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9294125
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9291097
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9290107
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.96
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9227628
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.96
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9294125
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.96
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9291097
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.96
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9290107

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9113228
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9113228
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9187070, upper bound: 1.9294125
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226475, upper bound: 1.9282253
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9218332
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280543, upper bound: 1.9218290
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9113228
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9113228
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9187070, upper bound: 1.9294125
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9226475, upper bound: 1.9282253
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9218332
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9280543, upper bound: 1.9218290
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120110
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121549
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9181879
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9178318
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9035558
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9088845, upper bound: 1.9034557
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9218290
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9172739
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177670, upper bound: 1.9270107
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120110
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121549
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9181879
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9178318
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9035558
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9088845, upper bound: 1.9034557
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9218290
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9172739
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9177670, upper bound: 1.9270107

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9162447
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9156593
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162275
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9156593
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9218290
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9168600
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183894, upper bound: 1.9172739
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9248490, upper bound: 1.9166054
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068470, upper bound: 1.9068932
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068470, upper bound: 1.9068642
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9162447
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9156593
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9156593
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9218290
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9168600
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9183894, upper bound: 1.9172739
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9248490, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9274808, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9068470, upper bound: 1.9068932
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9068470, upper bound: 1.9068642

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9039982
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183119, upper bound: 1.9166054
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9040293
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9039643
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113568, upper bound: 1.9034504
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113568, upper bound: 1.9034504
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115780, upper bound: 1.9034504
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115780, upper bound: 1.9034504
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.23 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9039982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9183119, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9040293
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9039643
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9113568, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9113568, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9115780, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -1.9115780, upper bound: 1.9034504

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.55 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.55
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.55
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
Binary search (step 7): status=Status.VERIFIED, low=0.1811080, high=0.1818182, mid=0.1811080, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 8) starts
Candidate diff: 0.1814631


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121549
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121549
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9173206
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9150670
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171814, upper bound: 1.9151243
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9173206
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9150670
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -1.9171814, upper bound: 1.9151243

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9099121
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9172919
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9171918
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9150670
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9072466
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9083481, upper bound: 1.9039643
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9069739
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.30 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9099121
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9172919
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9171918
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9150670
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9072466
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9083481, upper bound: 1.9039643
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -1.9035558, upper bound: 1.9069739

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098363, upper bound: 1.9161233
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9132033, upper bound: 1.9155210
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072462, upper bound: 1.9063323
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9084936, upper bound: 1.9090062
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070909
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.13 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9098363, upper bound: 1.9161233
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9132033, upper bound: 1.9155210
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9072462, upper bound: 1.9063323
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9084936, upper bound: 1.9090062
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.13
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070909
Binary search (step 8): status=Status.VERIFIED, low=0.1814631, high=0.1818182, mid=0.1814631, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 9) starts
Candidate diff: 0.1816406


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9291097
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9290107
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9291097
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9290107

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9280522
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9280522
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217026, upper bound: 1.9217609
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217026, upper bound: 1.9286693
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280543, upper bound: 1.9218332
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9290107
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9233958
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9280522
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9280522
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9217026, upper bound: 1.9217609
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9217026, upper bound: 1.9286693
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9280543, upper bound: 1.9218332
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9276301
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9290107
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9233958

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9188614
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280522
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9257274
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9063323, upper bound: 1.9072462
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9063323, upper bound: 1.9072462
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9035558
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9088845, upper bound: 1.9034557
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9249909, upper bound: 1.9276301
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9167719
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9218290
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9257274, upper bound: 1.9276301
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280543, upper bound: 1.9172739
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9221345
time: 0.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9188614
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280522
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9276301, upper bound: 1.9257274
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9063323, upper bound: 1.9072462
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9063323, upper bound: 1.9072462
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9035558
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9088845, upper bound: 1.9034557
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9249909, upper bound: 1.9276301
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9167719
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9218290
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9257274, upper bound: 1.9276301
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9280543, upper bound: 1.9172739
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -1.9280522, upper bound: 1.9221345

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9068470
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9068470
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9088371
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9037431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9257274
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9249909
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9123495, upper bound: 1.9034557
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9125640, upper bound: 1.9034557
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9166054
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177842, upper bound: 1.9217213
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9068282
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9034557
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9067042, upper bound: 1.9040293
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9083144, upper bound: 1.9039643
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082736, upper bound: 1.9071747
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9088759, upper bound: 1.9069739
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9088371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9037431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9257274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9249909
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9066648
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9126185, upper bound: 1.9065718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9123495, upper bound: 1.9034557
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9125640, upper bound: 1.9034557
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9177842, upper bound: 1.9217213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9068282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9126043, upper bound: 1.9034557
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9067042, upper bound: 1.9040293
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9083144, upper bound: 1.9039643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9082736, upper bound: 1.9071747
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.67
Output dim: 0, lower bound: -1.9088759, upper bound: 1.9069739

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9087141
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082624
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166411, upper bound: 1.9168328
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9169861
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181650, upper bound: 1.9248490
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9039532
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9087141
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9166411, upper bound: 1.9168328
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9169861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9181650, upper bound: 1.9248490
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9039532
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.16
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9107751
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
Binary search (step 9): status=Status.VERIFIED, low=0.1816406, high=0.1818182, mid=0.1816406, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 10) starts
Candidate diff: 0.1817294


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9309287
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9273981
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9226944
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9300951
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9309287
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9273981
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9226944
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9300951

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9309287
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9285226
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9118061
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9118065
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273981, upper bound: 1.9226944
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238470, upper bound: 1.9191455
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9121072
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9121919
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9309287
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9285226
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9118061
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9118065
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9273981, upper bound: 1.9226944
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9238470, upper bound: 1.9191455
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9121072
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.03
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9121919

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120110
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121549
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233958, upper bound: 1.9241714
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9236749
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9257909, upper bound: 1.9190868
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9257909, upper bound: 1.9170827
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9108930
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113228, upper bound: 1.9108930
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120110
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121549
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9233958, upper bound: 1.9241714
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9236749
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9257909, upper bound: 1.9190868
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9257909, upper bound: 1.9170827
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.12
Output dim: 0, lower bound: -1.9113228, upper bound: 1.9108930

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9239613
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9167719
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9234938
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9167719
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9098232
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9165176, upper bound: 1.9093852
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9239613
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9167719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9234938
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9167719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9098232
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -1.9165176, upper bound: 1.9093852

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9186595
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9237651
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9034557
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9063572
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9037431
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9186595
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9237651
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9034557
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9063572
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.12
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9037431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9058468
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9058468
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9058468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9058468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.58
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
Binary search (step 10): status=Status.VERIFIED, low=0.1817294, high=0.1818182, mid=0.1817294, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 11) starts
Candidate diff: 0.1817738


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9227628
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9294125
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9183080
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9227628
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9294125
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.00
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161542, upper bound: 1.9160467
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9173206, upper bound: 1.9160241
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9294125
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9273981
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9173206
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9192868, upper bound: 1.9161978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9110647
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9161542, upper bound: 1.9160467
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9173206, upper bound: 1.9160241
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9294125
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9273981
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9173206
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9192868, upper bound: 1.9161978
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9110647

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9107210, upper bound: 1.9160241
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9173166, upper bound: 1.9159419
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9183790
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121437, upper bound: 1.9170690
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9180951
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9178318
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9145044, upper bound: 1.9121437
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9159419, upper bound: 1.9173166
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9107210
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9079029, upper bound: 1.9051840
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183790, upper bound: 1.9106786
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160467, upper bound: 1.9107210
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9107210, upper bound: 1.9160241
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9173166, upper bound: 1.9159419
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9183790
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9121437, upper bound: 1.9170690
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9180951
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9178318
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9178318, upper bound: 1.9146498
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9145044, upper bound: 1.9121437
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9159419, upper bound: 1.9173166
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9107210
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9079029, upper bound: 1.9051840
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9183790, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -1.9160467, upper bound: 1.9107210

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9159337, upper bound: 1.9131966
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161233, upper bound: 1.9096788
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9103646
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9094921
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065706, upper bound: 1.9129553
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162957
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9160578
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9098232
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9061149
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082527, upper bound: 1.9090062
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9105563, upper bound: 1.9050081
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9129553, upper bound: 1.9050081
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9159337, upper bound: 1.9131966
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9161233, upper bound: 1.9096788
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9062904, upper bound: 1.9103646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9094921
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9065706, upper bound: 1.9129553
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9105563
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162957
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9160578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9122361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9162957, upper bound: 1.9098232
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9061149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9082527, upper bound: 1.9090062
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9105563, upper bound: 1.9050081
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.60
Output dim: 0, lower bound: -1.9129553, upper bound: 1.9050081
Binary search (step 11): status=Status.VERIFIED, low=0.1817738, high=0.1818182, mid=0.1817738, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 12) starts
Candidate diff: 0.1817960


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9287598
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9285347
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9226944
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9294125
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9287598
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9285347
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9226944
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -1.9226944, upper bound: 1.9294125

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233958, upper bound: 1.9287598
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9290107, upper bound: 1.9258611
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9286864, upper bound: 1.9172835
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190892, upper bound: 1.9280389
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9286693, upper bound: 1.9217026
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9217431
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9300951
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9238470
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9233958, upper bound: 1.9287598
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9290107, upper bound: 1.9258611
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9286864, upper bound: 1.9172835
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9190892, upper bound: 1.9280389
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9286693, upper bound: 1.9217026
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9217431
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9300951
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9238470

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233958, upper bound: 1.9191163
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9170827, upper bound: 1.9280588
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171152, upper bound: 1.9258611
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9290107, upper bound: 1.9251325
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9169116
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9286864, upper bound: 1.9172835
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169282, upper bound: 1.9280383
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190218, upper bound: 1.9249564
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273211, upper bound: 1.9177670
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121541, upper bound: 1.9113201
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9113201
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9172919
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9145044
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9171918
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160467, upper bound: 1.9161542
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9233958, upper bound: 1.9191163
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9170827, upper bound: 1.9280588
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9171152, upper bound: 1.9258611
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9290107, upper bound: 1.9251325
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9225726, upper bound: 1.9169116
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9286864, upper bound: 1.9172835
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9169282, upper bound: 1.9280383
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9190218, upper bound: 1.9249564
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9273211, upper bound: 1.9177670
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9121541, upper bound: 1.9113201
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9113201
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9172919
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9145044
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9160241, upper bound: 1.9171918
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.21
Output dim: 0, lower bound: -1.9160467, upper bound: 1.9161542

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9188544
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9183894
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9165597
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9148003
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9257274
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9228392
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9284977, upper bound: 1.9188968
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9170827, upper bound: 1.9249855
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9120149, upper bound: 1.9070909
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160707, upper bound: 1.9070909
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9128641, upper bound: 1.9070910
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098232, upper bound: 1.9098363
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072466, upper bound: 1.9160578
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9160578
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9182963, upper bound: 1.9232525
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9190120, upper bound: 1.9249564
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177468
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273211, upper bound: 1.9166371
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9166054
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9155742, upper bound: 1.9172725
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9106786
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9171903
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9107210
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9188544
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9183894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9165597
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9148003
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9257274
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9228392
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9284977, upper bound: 1.9188968
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9170827, upper bound: 1.9249855
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9120149, upper bound: 1.9070909
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9160707, upper bound: 1.9070909
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9128641, upper bound: 1.9070910
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9098232, upper bound: 1.9098363
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9072466, upper bound: 1.9160578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9160578
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9182963, upper bound: 1.9232525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9190120, upper bound: 1.9249564
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177468
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9273211, upper bound: 1.9166371
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9228392, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9273235, upper bound: 1.9166054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9155742, upper bound: 1.9172725
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9155832, upper bound: 1.9106786
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9106786, upper bound: 1.9171903
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -1.9140879, upper bound: 1.9107210

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9068470
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9068470
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166969, upper bound: 1.9183119
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9166054
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9125045
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9121872
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9083481
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9082598
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070910
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9131966
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162447
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162275
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172671, upper bound: 1.9230277
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181650, upper bound: 1.9248490
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177468, upper bound: 1.9166054
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177468
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9084923, upper bound: 1.9034504
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9085061, upper bound: 1.9034504
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9034504
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9083410, upper bound: 1.9034504
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9079305, upper bound: 1.9034504
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9058248
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9076755, upper bound: 1.9080810
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9061149
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9082527, upper bound: 1.9090062
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9166969, upper bound: 1.9183119
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9166054
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9125045
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9121872
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9083481
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9082598
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070910
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9131966
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9162275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9172671, upper bound: 1.9230277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9181650, upper bound: 1.9248490
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9177468, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177468
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9084923, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9085061, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9082598, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9083410, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9079305, upper bound: 1.9034504
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9058248
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9076755, upper bound: 1.9080810
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9061149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.12
Output dim: 0, lower bound: -1.9082527, upper bound: 1.9090062

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9034504
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9063431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034754, upper bound: 1.9037431
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9047733
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9047733
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9063431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034754, upper bound: 1.9037431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9047733
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9047733
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
Binary search (step 12): status=Status.VERIFIED, low=0.1817960, high=0.1818182, mid=0.1817960, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 13) starts
Candidate diff: 0.1818071


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9287598
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9285347
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9226944
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9227628, upper bound: 1.9300951
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -1.9285347, upper bound: 1.9287598
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -1.9287598, upper bound: 1.9285347
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -1.9294125, upper bound: 1.9226944
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -1.9227628, upper bound: 1.9300951

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9284977, upper bound: 1.9191191
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172007, upper bound: 1.9280588
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9171576
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171844, upper bound: 1.9163851
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9286693, upper bound: 1.9217026
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9217431
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217431, upper bound: 1.9265637
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9222794, upper bound: 1.9285013
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9284977, upper bound: 1.9191191
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9172007, upper bound: 1.9280588
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9171576
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9171844, upper bound: 1.9163851
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9286693, upper bound: 1.9217026
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9217431
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9217431, upper bound: 1.9265637
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9222794, upper bound: 1.9285013

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9188614
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217213, upper bound: 1.9184002
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9273235
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9072466, upper bound: 1.9171576
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9161935
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161233, upper bound: 1.9098363
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9160578
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9113205
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9118033, upper bound: 1.9113205
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183605, upper bound: 1.9216869
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9178584
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120257
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121536
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9090062
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9084936, upper bound: 1.9090062
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9188614
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9217213, upper bound: 1.9184002
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9273235
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9072466, upper bound: 1.9171576
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9161935
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9161233, upper bound: 1.9098363
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9160578
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9113205
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9118033, upper bound: 1.9113205
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9183605, upper bound: 1.9216869
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9178584
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120257
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121536
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9090062
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -1.9084936, upper bound: 1.9090062

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166411, upper bound: 1.9187800
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9186646
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9183894
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177842
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9083410
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082598
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9171576
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9169212
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9273879, upper bound: 1.9216787
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9183393
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9268152, upper bound: 1.9178584
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9177302
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9166411, upper bound: 1.9187800
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9186646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9183894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177842
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9256487
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9083410
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082598
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9171576
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9070926, upper bound: 1.9169212
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9273879, upper bound: 1.9216787
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9183393
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9268152, upper bound: 1.9178584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -1.9287806, upper bound: 1.9177302

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166080, upper bound: 1.9187708
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166411, upper bound: 1.9186373
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9186595
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9186447
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9054584
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9237651
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070909
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9148717
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9118065, upper bound: 1.9113201
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113201
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9230277, upper bound: 1.9172671
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9186595, upper bound: 1.9166054
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9114737, upper bound: 1.9108930
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9114737, upper bound: 1.9108930
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9097187, upper bound: 1.9050081
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9103646, upper bound: 1.9050081
time: 0.41 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9166080, upper bound: 1.9187708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9166411, upper bound: 1.9186373
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9186595
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9186447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9054584
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9237651
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070909
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9148717
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9118065, upper bound: 1.9113201
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113201
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9230277, upper bound: 1.9172671
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9186595, upper bound: 1.9166054
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9114737, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9114737, upper bound: 1.9108930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9097187, upper bound: 1.9050081
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.17
Output dim: 0, lower bound: -1.9103646, upper bound: 1.9050081

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9064408
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9058468
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9058468
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9057743
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9087141
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082624
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9037431, upper bound: 1.9034754
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9063431, upper bound: 1.9047025
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072411, upper bound: 1.9034504
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9064408
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9068642, upper bound: 1.9058468
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9058468
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9057743
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9087141
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082624
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9072411
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9037431, upper bound: 1.9034754
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9063431, upper bound: 1.9047025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9072411, upper bound: 1.9034504
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
Binary search (step 13): status=Status.VERIFIED, low=0.1818071, high=0.1818182, mid=0.1818071, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 14) starts
Candidate diff: 0.1818126


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
time: 0.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9121072
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9121919
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.09
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9121072
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.09
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9121919

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9151243, upper bound: 1.9171814
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9171576
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9151243, upper bound: 1.9171814
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150670, upper bound: 1.9171576
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150519, upper bound: 1.9132347
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098363, upper bound: 1.9165597
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9149894, upper bound: 1.9070910
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9160578
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -1.9150519, upper bound: 1.9132347
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -1.9098363, upper bound: 1.9165597
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -1.9149894, upper bound: 1.9070910
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.15
Output dim: 0, lower bound: -1.9122361, upper bound: 1.9160578
Binary search (step 14): status=Status.VERIFIED, low=0.1818126, high=0.1818182, mid=0.1818126, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 15) starts
Candidate diff: 0.1818154


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.67
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121549
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.98 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 1.98
Output dim: 0, lower bound: -1.9121919, upper bound: 1.9120110
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 1.98
Output dim: 0, lower bound: -1.9121072, upper bound: 1.9121549
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9150670
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171814, upper bound: 1.9151243
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.06 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9150670
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -1.9171814, upper bound: 1.9151243

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9161935, upper bound: 1.9150670
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9072466
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9163045, upper bound: 1.9151243
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9171814, upper bound: 1.9079385
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.19 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.9161935, upper bound: 1.9150670
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.9171576, upper bound: 1.9072466
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.9163045, upper bound: 1.9151243
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.19
Output dim: 0, lower bound: -1.9171814, upper bound: 1.9079385

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070909
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9039643
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9125045, upper bound: 1.9044216
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.55 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.55
Output dim: 0, lower bound: -1.9160578, upper bound: 1.9072466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.55
Output dim: 0, lower bound: -1.9070910, upper bound: 1.9070909
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.55
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9039643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.55
Output dim: 0, lower bound: -1.9125045, upper bound: 1.9044216
Binary search (step 15): status=Status.VERIFIED, low=0.1818154, high=0.1818182, mid=0.1818154, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 16) starts
Candidate diff: 0.1818168


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9309287
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9314710, upper bound: 1.9273981
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9121072
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9121919
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9238686, upper bound: 1.9309287
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9314710, upper bound: 1.9273981
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9121549, upper bound: 1.9121072
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.10
Output dim: 0, lower bound: -1.9120110, upper bound: 1.9121919

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9309287
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9238217, upper bound: 1.9285226
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9273879
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9242322
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9309287
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9238217, upper bound: 1.9285226
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9273879
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.01
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9242322

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9227115
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179098, upper bound: 1.9294125
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9237930, upper bound: 1.9226541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9282253
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070585, upper bound: 1.9150024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9091726, upper bound: 1.9150024
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9228392
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9233314
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9181005, upper bound: 1.9227115
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9179098, upper bound: 1.9294125
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9237930, upper bound: 1.9226541
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9191455, upper bound: 1.9282253
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9070585, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9091726, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9228392
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.97
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9233314

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9169482, upper bound: 1.9190330
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9169116, upper bound: 1.9169116
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9287806
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9286693
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233079, upper bound: 1.9222674
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9177302
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9181879
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9121437, upper bound: 1.9170690
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9228392
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9167719
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166802
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177670, upper bound: 1.9233314
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9169482, upper bound: 1.9190330
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9169116, upper bound: 1.9169116
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9287806
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9286693
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9233079, upper bound: 1.9222674
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9177302
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9137921, upper bound: 1.9181879
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9121437, upper bound: 1.9170690
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9228392
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9218290, upper bound: 1.9167719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166802
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -1.9177670, upper bound: 1.9233314

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166080, upper bound: 1.9187708
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166969, upper bound: 1.9183119
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120110
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121541
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9094921
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9080044
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113228
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113008, upper bound: 1.9113228
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9166054
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9154581
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9120149, upper bound: 1.9103000
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166969, upper bound: 1.9177402
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9228392
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9085061
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9084923
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9166080, upper bound: 1.9187708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9166969, upper bound: 1.9183119
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9120110
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9121541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9094921
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9050081, upper bound: 1.9080044
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9113008, upper bound: 1.9113228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9166054
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9099121, upper bound: 1.9154581
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9120149, upper bound: 1.9103000
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9166969, upper bound: 1.9177402
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9228392
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9085061
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.38
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9084923

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9054584
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9083410
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082598
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9068470
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058935
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9054584
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9083410
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.40
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9082598
Binary search (step 16): status=Status.VERIFIED, low=0.1818168, high=0.1818182, mid=0.1818168, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 17) starts
Candidate diff: 0.1818175


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.69
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9161978
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.93
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.93
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.93
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.93
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9161978

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9222794
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217431, upper bound: 1.9287806
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9295741
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9242322
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9192210, upper bound: 1.9145079
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9161978
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9222794
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9217431, upper bound: 1.9287806
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9295741
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9189582, upper bound: 1.9242322
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9192210, upper bound: 1.9145079
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9161978

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9179116, upper bound: 1.9221533
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9222674
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9182602, upper bound: 1.9274760
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9280543
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9265637, upper bound: 1.9188557
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9217026, upper bound: 1.9241929
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9169212, upper bound: 1.9129989
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9169009, upper bound: 1.9093852
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108377, upper bound: 1.9066018
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9112284, upper bound: 1.9098848
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9179116, upper bound: 1.9221533
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9222674
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9274808
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9182602, upper bound: 1.9274760
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9265637, upper bound: 1.9188557
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9217026, upper bound: 1.9241929
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9169212, upper bound: 1.9129989
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9169009, upper bound: 1.9093852
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9108377, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.36
Output dim: 0, lower bound: -1.9112284, upper bound: 1.9098848

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072166, upper bound: 1.9084936
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9072166, upper bound: 1.9084936
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9233079, upper bound: 1.9222674
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9219812
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9050319, upper bound: 1.9108040
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9167719
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9085366
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177842
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166802
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9241312
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9177982
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9072166, upper bound: 1.9084936
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9072166, upper bound: 1.9084936
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9233079, upper bound: 1.9222674
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9285013, upper bound: 1.9219812
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9115780
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9108040
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9050319, upper bound: 1.9108040
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9167719, upper bound: 1.9280543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9172739, upper bound: 1.9167719
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9085366
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177842
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166802
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9241312
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 0, lower bound: -1.9177302, upper bound: 1.9177982

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9186595
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9166054
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9070543, upper bound: 1.9050081
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9080810, upper bound: 1.9076755
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9083144
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9067042
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9034557
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177402
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113199
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113199
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9108930
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9221345, upper bound: 1.9186595
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9212821, upper bound: 1.9166054
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9070543, upper bound: 1.9050081
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9080810, upper bound: 1.9076755
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9083144
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9067042
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9034557, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9040293, upper bound: 1.9034557
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9177402
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9166054, upper bound: 1.9166054
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113199
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9113199
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9113205, upper bound: 1.9108930
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.39
Output dim: 0, lower bound: -1.9108930, upper bound: 1.9108930

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058468
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9058468
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9036108
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9036108
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Candidate
type: RSZ, layer: 3, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 6

### Candidate
type: RSZ, layer: 3, pos: 40

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9054584
time: 0.37 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9058468
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -1.9068932, upper bound: 1.9058468
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -1.9065718, upper bound: 1.9036108
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -1.9066648, upper bound: 1.9036108
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9034504
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.76
Output dim: 0, lower bound: -1.9034504, upper bound: 1.9054584
Binary search (step 17): status=Status.VERIFIED, low=0.1818175, high=0.1818182, mid=0.1818175, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1818174936554442
execution time: 1031.49 seconds
