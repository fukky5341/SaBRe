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
execution time: IAR + LP analysis = 1.33 + 1.09 = 2.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.9315851, upper bound: 1.9315851


# Binary Search by BASE starts (time budget: 1197.58 seconds, max iter: 100)

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
Binary search time: 43.81 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1153.77 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
time: 0.30 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978
time: 0.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9066018
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 0): status=Status.VERIFIED, low=0.0909091, high=0.1818182, mid=0.0909091, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 1) starts
Candidate diff: 0.1363636


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9161978, upper bound: 1.9195167
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9183080, upper bound: 1.9183853
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9183853, upper bound: 1.9183080
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -1.9195167, upper bound: 1.9161978

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 1): status=Status.VERIFIED, low=0.1363636, high=0.1818182, mid=0.1363636, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 2) starts
Candidate diff: 0.1590909


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.32 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9066018
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9098848
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9098848
Binary search (step 2): status=Status.VERIFIED, low=0.1590909, high=0.1818182, mid=0.1590909, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 3) starts
Candidate diff: 0.1704546


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.11
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 3): status=Status.VERIFIED, low=0.1704546, high=0.1818182, mid=0.1704546, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 4) starts
Candidate diff: 0.1761364


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 4): status=Status.VERIFIED, low=0.1761364, high=0.1818182, mid=0.1761364, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 5) starts
Candidate diff: 0.1789773


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295301
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.17
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 5): status=Status.VERIFIED, low=0.1789773, high=0.1818182, mid=0.1789773, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 6) starts
Candidate diff: 0.1803977


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.31 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9099032
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9098848
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9098848
Binary search (step 6): status=Status.VERIFIED, low=0.1803977, high=0.1818182, mid=0.1803977, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 7) starts
Candidate diff: 0.1811080


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 7): status=Status.VERIFIED, low=0.1811080, high=0.1818182, mid=0.1811080, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 8) starts
Candidate diff: 0.1814631


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295301
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.31 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 8): status=Status.VERIFIED, low=0.1814631, high=0.1818182, mid=0.1814631, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 9) starts
Candidate diff: 0.1816406


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 9): status=Status.VERIFIED, low=0.1816406, high=0.1818182, mid=0.1816406, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 10) starts
Candidate diff: 0.1817294


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9271448
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 10): status=Status.VERIFIED, low=0.1817294, high=0.1818182, mid=0.1817294, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 11) starts
Candidate diff: 0.1817738


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

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
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 11): status=Status.VERIFIED, low=0.1817738, high=0.1818182, mid=0.1817738, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 12) starts
Candidate diff: 0.1817960


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

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
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9098848
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.16
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9098848
Binary search (step 12): status=Status.VERIFIED, low=0.1817960, high=0.1818182, mid=0.1817960, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 13) starts
Candidate diff: 0.1818071


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9098848
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9098848
Binary search (step 13): status=Status.VERIFIED, low=0.1818071, high=0.1818182, mid=0.1818071, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 14) starts
Candidate diff: 0.1818126


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.14
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 14): status=Status.VERIFIED, low=0.1818126, high=0.1818182, mid=0.1818126, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 15) starts
Candidate diff: 0.1818154


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9309287
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -1.9309287, upper bound: 1.9314710

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
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
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

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
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9098848
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.12
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9098848
Binary search (step 15): status=Status.VERIFIED, low=0.1818154, high=0.1818182, mid=0.1818154, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 16) starts
Candidate diff: 0.1818168


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.20 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295301, upper bound: 1.9295608
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9271448, upper bound: 1.9295741
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.32 seconds

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
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

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
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.15
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 16): status=Status.VERIFIED, low=0.1818168, high=0.1818182, mid=0.1818168, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary search (step 17) starts
Candidate diff: 0.1818175


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

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
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295741, upper bound: 1.9271448
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.9295608, upper bound: 1.9295301
time: 0.39 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7438335, 1.5253004, -0.7438335, 1.5253004, -2.2691340, 2.2691340
1: -0.8104258, 2.3001223, -0.8104258, 2.3001223, -3.1105480, 3.1105480
2: -1.7379694, 1.6217314, -1.7379694, 1.6217314, -3.3597007, 3.3597007
3: -1.1569667, 3.6523972, -1.1569667, 3.6523972, -4.8093638, 4.8093638
4: -2.2834320, 1.8319958, -2.2834320, 1.8319958, -4.1154280, 4.1154280

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9098848, upper bound: 1.9150024
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9099032, upper bound: 1.9150024
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9116045
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9066018, upper bound: 1.9115883
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9066018
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9116045, upper bound: 1.9066018
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9115883, upper bound: 1.9099032
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.13
Output dim: 0, lower bound: -1.9150024, upper bound: 1.9098848
Binary search (step 17): status=Status.VERIFIED, low=0.1818175, high=0.1818182, mid=0.1818175, abs_max=2.269134044647217
rel_dist={0: [-1.9315851017326753, 1.9315851017326757]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1818174936554442
execution time: 288.44 seconds
