## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.045175422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553)
1: (-0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822)
2: (-0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808)
3: (-0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506)
4: (-0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249)

## BASE Result
execution time: IAR + LP analysis = 1.60 + 0.84 = 2.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0465726, upper bound: 0.0465726


# Binary Search by BASE starts (time budget: 1197.56 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=0.05155529826879501
rel_dist={0: [-0.04657254158633466, 0.04657254158633465]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=0.05155529826879501
rel_dist={0: [-0.04655466850732895, 0.046554668507328936]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=0.05155529826879501
rel_dist={0: [-0.04643723777492976, 0.04643723777492974]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0113636, mid=0.0113636, abs_max=0.05155529826879501
rel_dist={0: [-0.046257134670243685, 0.046257134670243644]}

## Binary search (step 4) starts
Candidate diff: 0.0056818


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0056818, mid=0.0056818, abs_max=0.05155529826879501
rel_dist={0: [-0.046105128879509696, 0.04610512888044604]}

## Binary search (step 5) starts
Candidate diff: 0.0028409


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0028409, mid=0.0028409, abs_max=0.05155529826879501
rel_dist={0: [-0.045911576120675604, 0.04591157612368746]}

## Binary search (step 6) starts
Candidate diff: 0.0014205


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0014205, mid=0.0014205, abs_max=0.05155529826879501
rel_dist={0: [-0.04568732200225989, 0.045687322002010186]}

## Binary search (step 7) starts
Candidate diff: 0.0007102


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007102, mid=0.0007102, abs_max=0.05155529826879501
rel_dist={0: [-0.045566110620896404, 0.045566110621249975]}

## Binary search (step 8) starts
Candidate diff: 0.0003551


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003551, mid=0.0003551, abs_max=0.05155529826879501
rel_dist={0: [-0.045504536918878964, 0.045504536919091856]}

## Binary search (step 9) starts
Candidate diff: 0.0001776


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001776, mid=0.0001776, abs_max=0.05155529826879501
rel_dist={0: [-0.04547234088213036, 0.045472340882226694]}

## Binary search (step 10) starts
Candidate diff: 0.0000888


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000888, mid=0.0000888, abs_max=0.05155529826879501
rel_dist={0: [-0.045455803727991226, 0.04545580372804092]}

## Binary search (step 11) starts
Candidate diff: 0.0000444


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000444, mid=0.0000444, abs_max=0.05155529826879501
rel_dist={0: [-0.045447531203799935, 0.04544753229716292]}

## Binary search (step 12) starts
Candidate diff: 0.0000222


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000222, mid=0.0000222, abs_max=0.05155529826879501
rel_dist={0: [-0.04544327253165521, 0.045443272531668105]}

## Binary search (step 13) starts
Candidate diff: 0.0000111


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000111, mid=0.0000111, abs_max=0.05155529826879501
rel_dist={0: [-0.045441142812514546, 0.0454411253591801]}

## Binary search (step 14) starts
Candidate diff: 0.0000055


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000055, mid=0.0000055, abs_max=0.05155529826879501
rel_dist={0: [-0.045440078245859696, 0.045440078245863005]}

## Binary search (step 15) starts
Candidate diff: 0.0000028


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000028, mid=0.0000028, abs_max=0.05155529826879501
rel_dist={0: [-0.04543954642833681, 0.04543954642833836]}

## Binary search (step 16) starts
Candidate diff: 0.0000014


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000014, mid=0.0000014, abs_max=0.05155529826879501
rel_dist={0: [-0.04543928115040508, 0.045439281150405936]}

## Binary search (step 17) starts
Candidate diff: 0.0000007


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000007, mid=0.0000007, abs_max=0.05155529826879501
rel_dist={0: [-0.045439151719456165, 0.04543914919181363]}

## Binary Search Result
Binary search time: 47.49 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1150.07 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 0): status=Status.VERIFIED, low=0.0909091, high=0.1818182, mid=0.0909091, abs_max=0.05155529826879501
rel_dist={0: [-0.04657254158633466, 0.04657254158633465]}

## Binary search (step 1) starts
Candidate diff: 0.1363636


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.54 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.54
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.54
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 1): status=Status.VERIFIED, low=0.1363636, high=0.1818182, mid=0.1363636, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 2) starts
Candidate diff: 0.1590909


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.58
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.58
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.68 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463382, upper bound: 0.0461227
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463383
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463382, upper bound: 0.0461214
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463382
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -0.0463382, upper bound: 0.0461227
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463383
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -0.0463382, upper bound: 0.0461214
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463382

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461419, upper bound: 0.0459139
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462220, upper bound: 0.0459846
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459815, upper bound: 0.0462217
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458379, upper bound: 0.0461293
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459846, upper bound: 0.0462220
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459139, upper bound: 0.0461419
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0461419, upper bound: 0.0459139
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0462220, upper bound: 0.0459846
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0459815, upper bound: 0.0462217
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0458379, upper bound: 0.0461293
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0459846, upper bound: 0.0462220
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -0.0459139, upper bound: 0.0461419

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453801
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451777
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453708, upper bound: 0.0450971
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451777
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453708, upper bound: 0.0450971
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450391, upper bound: 0.0452388
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454717, upper bound: 0.0450432
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454017, upper bound: 0.0452855
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454804, upper bound: 0.0450988
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455145
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452903, upper bound: 0.0451739
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452413, upper bound: 0.0454924
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452759, upper bound: 0.0454833
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453594
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451030, upper bound: 0.0451714
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451075
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451198
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453161, upper bound: 0.0450156
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451975, upper bound: 0.0451029
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451721, upper bound: 0.0451113
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0450902
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453518, upper bound: 0.0450883
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451976, upper bound: 0.0451029
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451721, upper bound: 0.0451113
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451388, upper bound: 0.0454924
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452762, upper bound: 0.0454832
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451388, upper bound: 0.0454924
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452762, upper bound: 0.0454832
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0453708
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451777, upper bound: 0.0451096
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450970, upper bound: 0.0453708
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0451096
time: 0.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451777
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0453708, upper bound: 0.0450971
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451777
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0453708, upper bound: 0.0450971
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450391, upper bound: 0.0452388
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0454717, upper bound: 0.0450432
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0454017, upper bound: 0.0452855
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0454804, upper bound: 0.0450988
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455145
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0452903, upper bound: 0.0451739
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0452413, upper bound: 0.0454924
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0452759, upper bound: 0.0454833
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453594
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451030, upper bound: 0.0451714
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451075
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0453161, upper bound: 0.0450156
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451975, upper bound: 0.0451029
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451721, upper bound: 0.0451113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0450902
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0453518, upper bound: 0.0450883
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451976, upper bound: 0.0451029
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451721, upper bound: 0.0451113
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451388, upper bound: 0.0454924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0452762, upper bound: 0.0454832
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451388, upper bound: 0.0454924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0452762, upper bound: 0.0454832
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0453708
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0451777, upper bound: 0.0451096
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450970, upper bound: 0.0453708
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.57
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0451096

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450987, upper bound: 0.0451753
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451683
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451461, upper bound: 0.0450928
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453692, upper bound: 0.0450971
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450836, upper bound: 0.0451596
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450838, upper bound: 0.0450710
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450113
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453349, upper bound: 0.0450084
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450098
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450207
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450712, upper bound: 0.0450432
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454717, upper bound: 0.0450426
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450169, upper bound: 0.0452321
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451273, upper bound: 0.0450883
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451033, upper bound: 0.0450889
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450426, upper bound: 0.0454667
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451599, upper bound: 0.0450402
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450961, upper bound: 0.0451661
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452830, upper bound: 0.0451240
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450862, upper bound: 0.0450998
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450769, upper bound: 0.0451116
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450848, upper bound: 0.0451397
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450752, upper bound: 0.0451652
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0453498
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450902, upper bound: 0.0451096
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451154, upper bound: 0.0450149
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453141, upper bound: 0.0450156
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451652, upper bound: 0.0450752
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450758, upper bound: 0.0450770
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451928, upper bound: 0.0450646
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453330, upper bound: 0.0450569
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451652, upper bound: 0.0450752
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451116, upper bound: 0.0450770
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450137, upper bound: 0.0454398
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450598, upper bound: 0.0450022
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450129, upper bound: 0.0454398
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452218, upper bound: 0.0450081
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451038, upper bound: 0.0454923
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451358, upper bound: 0.0451190
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450129, upper bound: 0.0454398
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452218, upper bound: 0.0450081
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0451028
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450767, upper bound: 0.0451182
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450569, upper bound: 0.0450839
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450646, upper bound: 0.0450837
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450084, upper bound: 0.0453349
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450113, upper bound: 0.0450202
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450987, upper bound: 0.0451753
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451683
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451461, upper bound: 0.0450928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0453692, upper bound: 0.0450971
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450836, upper bound: 0.0451596
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450838, upper bound: 0.0450710
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450113
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0453349, upper bound: 0.0450084
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450098
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450207
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450712, upper bound: 0.0450432
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0454717, upper bound: 0.0450426
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450169, upper bound: 0.0452321
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451273, upper bound: 0.0450883
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451033, upper bound: 0.0450889
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450426, upper bound: 0.0454667
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451599, upper bound: 0.0450402
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450961, upper bound: 0.0451661
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0452830, upper bound: 0.0451240
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450862, upper bound: 0.0450998
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450769, upper bound: 0.0451116
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450848, upper bound: 0.0451397
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450752, upper bound: 0.0451652
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0453498
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450902, upper bound: 0.0451096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451154, upper bound: 0.0450149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0453141, upper bound: 0.0450156
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451652, upper bound: 0.0450752
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450758, upper bound: 0.0450770
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451928, upper bound: 0.0450646
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0453330, upper bound: 0.0450569
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451652, upper bound: 0.0450752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451116, upper bound: 0.0450770
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450137, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450598, upper bound: 0.0450022
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450129, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0452218, upper bound: 0.0450081
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451038, upper bound: 0.0454923
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0451358, upper bound: 0.0451190
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450129, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0452218, upper bound: 0.0450081
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0451028
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450767, upper bound: 0.0451182
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450569, upper bound: 0.0450839
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450646, upper bound: 0.0450837
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450084, upper bound: 0.0453349
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.13
Output dim: 0, lower bound: -0.0450113, upper bound: 0.0450202

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450113
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453334, upper bound: 0.0450084
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450746, upper bound: 0.0450023
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453334, upper bound: 0.0450084
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454289, upper bound: 0.0450075
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450149, upper bound: 0.0449902
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450123, upper bound: 0.0450109
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452926, upper bound: 0.0449997
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450070, upper bound: 0.0454196
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450273, upper bound: 0.0453255
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450241, upper bound: 0.0450517
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452296, upper bound: 0.0450124
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0450928
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450767, upper bound: 0.0451098
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450186, upper bound: 0.0449906
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453042, upper bound: 0.0449913
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450849, upper bound: 0.0450546
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450667, upper bound: 0.0450567
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450541, upper bound: 0.0450541
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450541, upper bound: 0.0450541
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453925
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449997, upper bound: 0.0452926
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450127, upper bound: 0.0454398
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450126, upper bound: 0.0450118
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449950, upper bound: 0.0450007
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449862, upper bound: 0.0450007
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450135, upper bound: 0.0454398
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450264, upper bound: 0.0449896
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450127, upper bound: 0.0454398
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450126, upper bound: 0.0450118
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449868, upper bound: 0.0449913
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452151, upper bound: 0.0449910
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449952, upper bound: 0.0450146
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449930, upper bound: 0.0450330
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450113
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0453334, upper bound: 0.0450084
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450746, upper bound: 0.0450023
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0453334, upper bound: 0.0450084
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0454289, upper bound: 0.0450075
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450149, upper bound: 0.0449902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450123, upper bound: 0.0450109
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0452926, upper bound: 0.0449997
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450070, upper bound: 0.0454196
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450273, upper bound: 0.0453255
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450241, upper bound: 0.0450517
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0452296, upper bound: 0.0450124
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0450928
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450767, upper bound: 0.0451098
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450186, upper bound: 0.0449906
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0453042, upper bound: 0.0449913
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450849, upper bound: 0.0450546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450667, upper bound: 0.0450567
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450541, upper bound: 0.0450541
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450541, upper bound: 0.0450541
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453925
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449997, upper bound: 0.0452926
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450127, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450126, upper bound: 0.0450118
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449950, upper bound: 0.0450007
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449862, upper bound: 0.0450007
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450135, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450264, upper bound: 0.0449896
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450127, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0450126, upper bound: 0.0450118
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449868, upper bound: 0.0449913
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0452151, upper bound: 0.0449910
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449952, upper bound: 0.0450146
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.94
Output dim: 0, lower bound: -0.0449930, upper bound: 0.0450330

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450330, upper bound: 0.0449931
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450146, upper bound: 0.0449952
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450330, upper bound: 0.0449931
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450146, upper bound: 0.0449952
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452926, upper bound: 0.0449997
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453925, upper bound: 0.0449832
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449893, upper bound: 0.0449962
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450115, upper bound: 0.0449975
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452926, upper bound: 0.0449997
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450005, upper bound: 0.0449902
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449999, upper bound: 0.0450270
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0452925
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450008, upper bound: 0.0452964
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450109, upper bound: 0.0450025
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449984, upper bound: 0.0450026
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450196, upper bound: 0.0449902
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0449902
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453925
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449785, upper bound: 0.0450289
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449855, upper bound: 0.0449817
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449772, upper bound: 0.0450073
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449835, upper bound: 0.0453948
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450009, upper bound: 0.0452965
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453925
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449997, upper bound: 0.0452926
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449898, upper bound: 0.0450256
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449885, upper bound: 0.0450509
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451383, upper bound: 0.0449910
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452128, upper bound: 0.0449822
time: 0.30 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450330, upper bound: 0.0449931
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450146, upper bound: 0.0449952
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450330, upper bound: 0.0449931
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450146, upper bound: 0.0449952
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0452926, upper bound: 0.0449997
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0453925, upper bound: 0.0449832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449893, upper bound: 0.0449962
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450115, upper bound: 0.0449975
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0452926, upper bound: 0.0449997
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450005, upper bound: 0.0449902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449999, upper bound: 0.0450270
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0452925
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450008, upper bound: 0.0452964
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450109, upper bound: 0.0450025
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449984, upper bound: 0.0450026
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450196, upper bound: 0.0449902
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0449902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453925
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449785, upper bound: 0.0450289
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449855, upper bound: 0.0449817
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449772, upper bound: 0.0450073
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449835, upper bound: 0.0453948
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0450009, upper bound: 0.0452965
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453925
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449997, upper bound: 0.0452926
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449898, upper bound: 0.0450256
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0449885, upper bound: 0.0450509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0451383, upper bound: 0.0449910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.41
Output dim: 0, lower bound: -0.0452128, upper bound: 0.0449822

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449805
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450034, upper bound: 0.0449751
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449739, upper bound: 0.0449777
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449948, upper bound: 0.0449762
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449817, upper bound: 0.0449805
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449805, upper bound: 0.0449817
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0450072
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449778, upper bound: 0.0449937
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449727, upper bound: 0.0450178
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449762, upper bound: 0.0449948
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449777, upper bound: 0.0449739
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449751, upper bound: 0.0450034
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449939
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449728, upper bound: 0.0450179
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449762, upper bound: 0.0449948
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449805, upper bound: 0.0449817
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449761, upper bound: 0.0450073
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
time: 0.33 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449805
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0450034, upper bound: 0.0449751
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449739, upper bound: 0.0449777
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449948, upper bound: 0.0449762
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449817, upper bound: 0.0449805
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449805, upper bound: 0.0449817
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0450072
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449778, upper bound: 0.0449937
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449727, upper bound: 0.0450178
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449762, upper bound: 0.0449948
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449777, upper bound: 0.0449739
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449751, upper bound: 0.0450034
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449939
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449728, upper bound: 0.0450179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449762, upper bound: 0.0449948
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449805, upper bound: 0.0449817
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449761, upper bound: 0.0450073
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.33
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
Binary search (step 2): status=Status.VERIFIED, low=0.1590909, high=0.1818182, mid=0.1590909, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042006]}

## Binary search (step 3) starts
Candidate diff: 0.1704546


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.67 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0461227
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463383
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462664
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462663, upper bound: 0.0461859
time: 0.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0461227
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463383
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462664
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0462663, upper bound: 0.0461859

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461419, upper bound: 0.0459139
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462220, upper bound: 0.0459846
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461293, upper bound: 0.0458379
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459846, upper bound: 0.0462220
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0461419, upper bound: 0.0459139
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0462220, upper bound: 0.0459846
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0461293, upper bound: 0.0458379
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0459846, upper bound: 0.0462220
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453522, upper bound: 0.0451668
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453544, upper bound: 0.0450825
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0451315
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453446, upper bound: 0.0450393
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454832, upper bound: 0.0452762
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454924, upper bound: 0.0451388
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451978, upper bound: 0.0451029
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451726, upper bound: 0.0451113
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452413, upper bound: 0.0454924
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452759, upper bound: 0.0454833
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453161
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0450419
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452188, upper bound: 0.0455145
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452903, upper bound: 0.0451739
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453594
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451030, upper bound: 0.0451714
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453161, upper bound: 0.0450156
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451714, upper bound: 0.0451030
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453594, upper bound: 0.0451053
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452194, upper bound: 0.0455144
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452906, upper bound: 0.0451440
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451726
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451978
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450402, upper bound: 0.0452386
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450432
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453447
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451314, upper bound: 0.0450419
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451975, upper bound: 0.0451029
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450944, upper bound: 0.0451113
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0453447
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451314, upper bound: 0.0450419
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0453522, upper bound: 0.0451668
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0453544, upper bound: 0.0450825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0451315
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0453446, upper bound: 0.0450393
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0454832, upper bound: 0.0452762
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0454924, upper bound: 0.0451388
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451978, upper bound: 0.0451029
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451726, upper bound: 0.0451113
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0452413, upper bound: 0.0454924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0452759, upper bound: 0.0454833
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453161
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0450419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0452188, upper bound: 0.0455145
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0452903, upper bound: 0.0451739
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453594
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451030, upper bound: 0.0451714
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0453161, upper bound: 0.0450156
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451714, upper bound: 0.0451030
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0453594, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0452194, upper bound: 0.0455144
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0452906, upper bound: 0.0451440
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451726
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450402, upper bound: 0.0452386
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450432
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451314, upper bound: 0.0450419
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451975, upper bound: 0.0451029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450944, upper bound: 0.0451113
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0453447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -0.0451314, upper bound: 0.0450419

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450974, upper bound: 0.0451645
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453506, upper bound: 0.0451568
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450046, upper bound: 0.0450019
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453189, upper bound: 0.0450008
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453155, upper bound: 0.0450100
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453189, upper bound: 0.0450008
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450081, upper bound: 0.0452218
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450129
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450022, upper bound: 0.0450598
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450137
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450098
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450801, upper bound: 0.0450157
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450136, upper bound: 0.0454397
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451854, upper bound: 0.0450099
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450125, upper bound: 0.0454290
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452215, upper bound: 0.0450094
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449897, upper bound: 0.0452972
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449904, upper bound: 0.0451367
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450426, upper bound: 0.0454667
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451599, upper bound: 0.0450402
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452391, upper bound: 0.0451545
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452735, upper bound: 0.0450943
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450149, upper bound: 0.0453141
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0450419
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450297, upper bound: 0.0450059
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450131, upper bound: 0.0450059
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453141, upper bound: 0.0450156
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450987, upper bound: 0.0454804
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452131, upper bound: 0.0454018
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0450712
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452365, upper bound: 0.0450308
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0451273
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450907, upper bound: 0.0451644
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0452320
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450201, upper bound: 0.0450098
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454290, upper bound: 0.0450126
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454397, upper bound: 0.0450136
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450376, upper bound: 0.0453430
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0450869
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451652, upper bound: 0.0450752
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451116, upper bound: 0.0450770
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450084, upper bound: 0.0453349
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450199
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450974, upper bound: 0.0451645
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0453506, upper bound: 0.0451568
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450046, upper bound: 0.0450019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0453189, upper bound: 0.0450008
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0453155, upper bound: 0.0450100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0453189, upper bound: 0.0450008
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450081, upper bound: 0.0452218
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450129
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450022, upper bound: 0.0450598
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450098
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450801, upper bound: 0.0450157
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450136, upper bound: 0.0454397
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0451854, upper bound: 0.0450099
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450125, upper bound: 0.0454290
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0452215, upper bound: 0.0450094
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0449897, upper bound: 0.0452972
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0449904, upper bound: 0.0451367
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450426, upper bound: 0.0454667
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0451599, upper bound: 0.0450402
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0452391, upper bound: 0.0451545
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0452735, upper bound: 0.0450943
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450149, upper bound: 0.0453141
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0450419
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450297, upper bound: 0.0450059
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450131, upper bound: 0.0450059
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0453141, upper bound: 0.0450156
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450987, upper bound: 0.0454804
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0452131, upper bound: 0.0454018
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0450712
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0452365, upper bound: 0.0450308
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0451273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450907, upper bound: 0.0451644
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0452320
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450201, upper bound: 0.0450098
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0454290, upper bound: 0.0450126
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0454397, upper bound: 0.0450136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450376, upper bound: 0.0453430
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0450869
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0451652, upper bound: 0.0450752
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0451116, upper bound: 0.0450770
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450084, upper bound: 0.0453349
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450199

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450955, upper bound: 0.0450710
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450817, upper bound: 0.0450748
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450637, upper bound: 0.0450008
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453172, upper bound: 0.0449963
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450043, upper bound: 0.0449847
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449930, upper bound: 0.0449853
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450005, upper bound: 0.0449826
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449908, upper bound: 0.0449826
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0452151
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449868
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450117, upper bound: 0.0450127
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450127
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450269, upper bound: 0.0449891
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0449949
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449949, upper bound: 0.0450059
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449891, upper bound: 0.0450269
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0450099
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451832, upper bound: 0.0450077
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0450253
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449887, upper bound: 0.0450507
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449865, upper bound: 0.0449931
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452149, upper bound: 0.0449914
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0452872
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449954
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450134, upper bound: 0.0454397
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450114, upper bound: 0.0454290
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450134, upper bound: 0.0450995
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451832, upper bound: 0.0450077
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450703, upper bound: 0.0450744
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452659, upper bound: 0.0450793
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450059
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450291
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451336, upper bound: 0.0449904
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452952, upper bound: 0.0449897
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0450909
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450880, upper bound: 0.0451273
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450943, upper bound: 0.0453657
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451973, upper bound: 0.0453721
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449906, upper bound: 0.0449906
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452298, upper bound: 0.0450062
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450149, upper bound: 0.0449984
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450122, upper bound: 0.0450109
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450118, upper bound: 0.0450104
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454290, upper bound: 0.0450115
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450995, upper bound: 0.0450134
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454397, upper bound: 0.0450135
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450082, upper bound: 0.0450319
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450444
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449773, upper bound: 0.0453090
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449679, upper bound: 0.0453058
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450955, upper bound: 0.0450710
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450817, upper bound: 0.0450748
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450637, upper bound: 0.0450008
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0453172, upper bound: 0.0449963
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450043, upper bound: 0.0449847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449930, upper bound: 0.0449853
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450005, upper bound: 0.0449826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449908, upper bound: 0.0449826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0452151
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449868
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450117, upper bound: 0.0450127
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450127
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450269, upper bound: 0.0449891
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0449949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449949, upper bound: 0.0450059
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449891, upper bound: 0.0450269
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0450099
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0451832, upper bound: 0.0450077
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0450253
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449887, upper bound: 0.0450507
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449865, upper bound: 0.0449931
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0452149, upper bound: 0.0449914
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0452872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449954
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450134, upper bound: 0.0454397
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450114, upper bound: 0.0454290
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450134, upper bound: 0.0450995
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0451832, upper bound: 0.0450077
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450703, upper bound: 0.0450744
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0452659, upper bound: 0.0450793
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450059
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450291
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0451336, upper bound: 0.0449904
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0452952, upper bound: 0.0449897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0450909
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450880, upper bound: 0.0451273
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450943, upper bound: 0.0453657
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0451973, upper bound: 0.0453721
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449906, upper bound: 0.0449906
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0452298, upper bound: 0.0450062
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450149, upper bound: 0.0449984
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450122, upper bound: 0.0450109
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450118, upper bound: 0.0450104
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0454290, upper bound: 0.0450115
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450995, upper bound: 0.0450134
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0454397, upper bound: 0.0450135
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450082, upper bound: 0.0450319
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450444
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449773, upper bound: 0.0453090
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -0.0449679, upper bound: 0.0453058

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0449826
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449822, upper bound: 0.0452128
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0451383
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450508, upper bound: 0.0449886
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450256, upper bound: 0.0449899
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449944, upper bound: 0.0449923
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449857, upper bound: 0.0449923
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451378, upper bound: 0.0449914
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452125, upper bound: 0.0449845
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449670
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449901
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453915
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0452925
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449898, upper bound: 0.0450253
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449885, upper bound: 0.0450507
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449793, upper bound: 0.0449888
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451769, upper bound: 0.0449863
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449982, upper bound: 0.0449894
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452125, upper bound: 0.0449845
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0449826
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450733, upper bound: 0.0450790
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450637, upper bound: 0.0450957
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450009, upper bound: 0.0452965
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451383, upper bound: 0.0449910
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450109, upper bound: 0.0450025
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0450026
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450507, upper bound: 0.0449886
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450253, upper bound: 0.0449899
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450269, upper bound: 0.0449889
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0449911
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449773, upper bound: 0.0453077
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449771, upper bound: 0.0450508
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449679, upper bound: 0.0453047
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449979
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0449826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449822, upper bound: 0.0452128
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0451383
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450508, upper bound: 0.0449886
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450256, upper bound: 0.0449899
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449944, upper bound: 0.0449923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449857, upper bound: 0.0449923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0451378, upper bound: 0.0449914
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0452125, upper bound: 0.0449845
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453915
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0452925
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449898, upper bound: 0.0450253
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449885, upper bound: 0.0450507
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449793, upper bound: 0.0449888
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0451769, upper bound: 0.0449863
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449982, upper bound: 0.0449894
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0452125, upper bound: 0.0449845
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0449826
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450733, upper bound: 0.0450790
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450637, upper bound: 0.0450957
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450009, upper bound: 0.0452965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0451383, upper bound: 0.0449910
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450109, upper bound: 0.0450025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0450026
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450507, upper bound: 0.0449886
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450253, upper bound: 0.0449899
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450269, upper bound: 0.0449889
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0449911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449773, upper bound: 0.0453077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449771, upper bound: 0.0450508
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449679, upper bound: 0.0453047
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449979

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449798, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449793, upper bound: 0.0449848
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449762, upper bound: 0.0449948
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449805, upper bound: 0.0449817
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449761, upper bound: 0.0450072
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449858, upper bound: 0.0449775
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449729, upper bound: 0.0449779
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449939
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449727, upper bound: 0.0450179
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449702, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449900
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449736, upper bound: 0.0449772
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449710, upper bound: 0.0449957
time: 0.33 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449798, upper bound: 0.0449670
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449793, upper bound: 0.0449848
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449762, upper bound: 0.0449948
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449805, upper bound: 0.0449817
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449761, upper bound: 0.0450072
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449858, upper bound: 0.0449775
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449729, upper bound: 0.0449779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449939
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449727, upper bound: 0.0450179
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449702, upper bound: 0.0449670
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449900
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449736, upper bound: 0.0449772
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.26
Output dim: 0, lower bound: -0.0449710, upper bound: 0.0449957
Binary search (step 3): status=Status.VERIFIED, low=0.1704546, high=0.1818182, mid=0.1704546, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 4) starts
Candidate diff: 0.1761364


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 4): status=Status.VERIFIED, low=0.1761364, high=0.1818182, mid=0.1761364, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 5) starts
Candidate diff: 0.1789773


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 5): status=Status.VERIFIED, low=0.1789773, high=0.1818182, mid=0.1789773, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 6) starts
Candidate diff: 0.1803977


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 6): status=Status.VERIFIED, low=0.1803977, high=0.1818182, mid=0.1803977, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 7) starts
Candidate diff: 0.1811080


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.72 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462663
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462664, upper bound: 0.0461733
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.66 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462664
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462663, upper bound: 0.0461859
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462663
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -0.0462664, upper bound: 0.0461733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462664
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 0, lower bound: -0.0462663, upper bound: 0.0461859

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0455767
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453614
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453614
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0455767
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453614
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453614
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -0.0455767, upper bound: 0.0453801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451777
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453708, upper bound: 0.0450971
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452188, upper bound: 0.0455145
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452903, upper bound: 0.0451739
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451316, upper bound: 0.0450944
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451179, upper bound: 0.0450973
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0454667
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452386, upper bound: 0.0450402
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451978, upper bound: 0.0451029
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451726, upper bound: 0.0451113
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453594
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451030, upper bound: 0.0451714
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454832, upper bound: 0.0452762
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454924, upper bound: 0.0451388
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451075
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451198
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0450912
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451075, upper bound: 0.0450915
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451726
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451978
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451994, upper bound: 0.0450803
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453426, upper bound: 0.0450718
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451726
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451978
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454016, upper bound: 0.0452853
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454804, upper bound: 0.0450987
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0453447
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451314, upper bound: 0.0450419
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454016, upper bound: 0.0452853
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454804, upper bound: 0.0450987
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451179
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451316
time: 0.27 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451777
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0453708, upper bound: 0.0450971
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0452188, upper bound: 0.0455145
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0452903, upper bound: 0.0451739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451316, upper bound: 0.0450944
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451179, upper bound: 0.0450973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0454667
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0452386, upper bound: 0.0450402
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451978, upper bound: 0.0451029
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451726, upper bound: 0.0451113
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453594
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451030, upper bound: 0.0451714
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0454832, upper bound: 0.0452762
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0454924, upper bound: 0.0451388
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451075
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0450912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451075, upper bound: 0.0450915
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451726
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451978
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451994, upper bound: 0.0450803
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0453426, upper bound: 0.0450718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451726
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0454016, upper bound: 0.0452853
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0454804, upper bound: 0.0450987
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0453447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0451314, upper bound: 0.0450419
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0454016, upper bound: 0.0452853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0454804, upper bound: 0.0450987
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451179
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451316

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451080, upper bound: 0.0451753
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451683
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451461, upper bound: 0.0450928
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453692, upper bound: 0.0450971
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451061, upper bound: 0.0451721
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451009, upper bound: 0.0451976
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452391, upper bound: 0.0451545
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452735, upper bound: 0.0450943
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450070, upper bound: 0.0454196
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450273, upper bound: 0.0453255
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451854, upper bound: 0.0450099
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452215, upper bound: 0.0450094
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450098
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450801, upper bound: 0.0450157
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453141
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450263, upper bound: 0.0450419
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450081, upper bound: 0.0452218
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450129
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451107, upper bound: 0.0450769
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450945, upper bound: 0.0450862
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450769, upper bound: 0.0451107
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450752, upper bound: 0.0451655
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450944, upper bound: 0.0450677
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450812, upper bound: 0.0450682
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451514, upper bound: 0.0450718
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453406, upper bound: 0.0450718
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0451273
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450907, upper bound: 0.0451644
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0452320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453255, upper bound: 0.0450273
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450201, upper bound: 0.0450098
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449906, upper bound: 0.0450070
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450084, upper bound: 0.0453349
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450199
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0452320
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453255, upper bound: 0.0450273
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454506, upper bound: 0.0450748
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454546, upper bound: 0.0450732
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451080, upper bound: 0.0451753
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451096, upper bound: 0.0451683
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451461, upper bound: 0.0450928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0453692, upper bound: 0.0450971
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451061, upper bound: 0.0451721
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451009, upper bound: 0.0451976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0452391, upper bound: 0.0451545
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0452735, upper bound: 0.0450943
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450070, upper bound: 0.0454196
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450273, upper bound: 0.0453255
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451854, upper bound: 0.0450099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0452215, upper bound: 0.0450094
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450801, upper bound: 0.0450157
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453141
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450263, upper bound: 0.0450419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450081, upper bound: 0.0452218
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0454398, upper bound: 0.0450129
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451107, upper bound: 0.0450769
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450945, upper bound: 0.0450862
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450769, upper bound: 0.0451107
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450752, upper bound: 0.0451655
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450944, upper bound: 0.0450677
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450812, upper bound: 0.0450682
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0451514, upper bound: 0.0450718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0453406, upper bound: 0.0450718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450883, upper bound: 0.0451273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450907, upper bound: 0.0451644
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0452320
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0453255, upper bound: 0.0450273
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450201, upper bound: 0.0450098
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0449906, upper bound: 0.0450070
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450084, upper bound: 0.0453349
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450199
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0452320
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0453255, upper bound: 0.0450273
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0454506, upper bound: 0.0450748
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 0, lower bound: -0.0454546, upper bound: 0.0450732

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450113
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453334, upper bound: 0.0450084
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450144, upper bound: 0.0450783
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450118, upper bound: 0.0450316
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450702, upper bound: 0.0451466
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452328, upper bound: 0.0451004
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450569, upper bound: 0.0450685
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452659, upper bound: 0.0450793
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453915
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0453901
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450103, upper bound: 0.0450211
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450021, upper bound: 0.0450452
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449837, upper bound: 0.0449929
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451791, upper bound: 0.0449911
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449950, upper bound: 0.0450007
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449879, upper bound: 0.0450007
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449897, upper bound: 0.0452953
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449897, upper bound: 0.0451337
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0452151
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449913, upper bound: 0.0449868
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450888, upper bound: 0.0450676
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450724, upper bound: 0.0450676
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450062, upper bound: 0.0452297
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0451537
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452963, upper bound: 0.0450008
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452925, upper bound: 0.0449996
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0453090
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449839, upper bound: 0.0453058
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450062, upper bound: 0.0452297
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0451537
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450452, upper bound: 0.0450022
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450210, upper bound: 0.0450103
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449931, upper bound: 0.0449865
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453901, upper bound: 0.0449827
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450847, upper bound: 0.0450639
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450540, upper bound: 0.0450666
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450202, upper bound: 0.0450113
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0453334, upper bound: 0.0450084
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450144, upper bound: 0.0450783
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450118, upper bound: 0.0450316
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450702, upper bound: 0.0451466
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0452328, upper bound: 0.0451004
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450569, upper bound: 0.0450685
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0452659, upper bound: 0.0450793
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449831, upper bound: 0.0453915
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0453901
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450103, upper bound: 0.0450211
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450021, upper bound: 0.0450452
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449837, upper bound: 0.0449929
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0451791, upper bound: 0.0449911
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449950, upper bound: 0.0450007
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449879, upper bound: 0.0450007
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449897, upper bound: 0.0452953
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449897, upper bound: 0.0451337
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0452151
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449913, upper bound: 0.0449868
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450888, upper bound: 0.0450676
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450724, upper bound: 0.0450676
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450062, upper bound: 0.0452297
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0451537
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0452963, upper bound: 0.0450008
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0452925, upper bound: 0.0449996
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0453090
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449839, upper bound: 0.0453058
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450062, upper bound: 0.0452297
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450172, upper bound: 0.0451537
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450452, upper bound: 0.0450022
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450210, upper bound: 0.0450103
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0449931, upper bound: 0.0449865
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0453901, upper bound: 0.0449827
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450847, upper bound: 0.0450639
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.99
Output dim: 0, lower bound: -0.0450540, upper bound: 0.0450666

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453047, upper bound: 0.0449839
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453077, upper bound: 0.0449773
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449974, upper bound: 0.0450279
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451769, upper bound: 0.0449863
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449894
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452125, upper bound: 0.0449845
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449763, upper bound: 0.0449948
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449777, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449751, upper bound: 0.0450024
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449861, upper bound: 0.0449775
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449733, upper bound: 0.0449781
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449996
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449822, upper bound: 0.0452128
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0451383
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449671
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450026, upper bound: 0.0449984
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450024, upper bound: 0.0450109
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450178, upper bound: 0.0449728
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449937, upper bound: 0.0449778
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450278, upper bound: 0.0449979
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452925, upper bound: 0.0449996
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449702, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449688, upper bound: 0.0449900
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449736, upper bound: 0.0449772
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449959
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449822, upper bound: 0.0452126
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449863, upper bound: 0.0451769
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449671
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453901, upper bound: 0.0449827
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0453047, upper bound: 0.0449839
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0453077, upper bound: 0.0449773
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449974, upper bound: 0.0450279
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0451769, upper bound: 0.0449863
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449894
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0452125, upper bound: 0.0449845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449782, upper bound: 0.0449670
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449763, upper bound: 0.0449948
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449777, upper bound: 0.0449670
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449751, upper bound: 0.0450024
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449861, upper bound: 0.0449775
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449733, upper bound: 0.0449781
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449822, upper bound: 0.0452128
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0451383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449671
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0450026, upper bound: 0.0449984
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0450024, upper bound: 0.0450109
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0450178, upper bound: 0.0449728
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449937, upper bound: 0.0449778
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0450278, upper bound: 0.0449979
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0452925, upper bound: 0.0449996
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449702, upper bound: 0.0449670
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449688, upper bound: 0.0449900
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449736, upper bound: 0.0449772
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449822, upper bound: 0.0452126
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449863, upper bound: 0.0451769
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449671
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.28
Output dim: 0, lower bound: -0.0453901, upper bound: 0.0449827

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449957, upper bound: 0.0449710
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449771, upper bound: 0.0449736
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449900, upper bound: 0.0449688
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449702
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449858, upper bound: 0.0449775
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449729, upper bound: 0.0449779
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449798, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449793, upper bound: 0.0449848
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450034, upper bound: 0.0449751
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449739, upper bound: 0.0449777
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449816, upper bound: 0.0449805
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449798, upper bound: 0.0449670
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449792, upper bound: 0.0449848
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449729
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449775, upper bound: 0.0449858
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450024, upper bound: 0.0449751
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449777
time: 0.29 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449957, upper bound: 0.0449710
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449771, upper bound: 0.0449736
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449900, upper bound: 0.0449688
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449702
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449858, upper bound: 0.0449775
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449729, upper bound: 0.0449779
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449848, upper bound: 0.0449793
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449798
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449798, upper bound: 0.0449670
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449793, upper bound: 0.0449848
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0450034, upper bound: 0.0449751
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449739, upper bound: 0.0449777
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449816, upper bound: 0.0449805
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449798, upper bound: 0.0449670
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449792, upper bound: 0.0449848
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449729
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449775, upper bound: 0.0449858
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0450024, upper bound: 0.0449751
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.77
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449777
Binary search (step 7): status=Status.VERIFIED, low=0.1811080, high=0.1818182, mid=0.1811080, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 8) starts
Candidate diff: 0.1814631


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.56
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.56
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.65 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455822
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455822
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.64 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0461214
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463382
time: 0.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455822
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455822
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0461214
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.0461214, upper bound: 0.0463382

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455178, upper bound: 0.0452979
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
time: 0.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452979
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455179, upper bound: 0.0452976
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459846, upper bound: 0.0462220
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0459139, upper bound: 0.0461419
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0455178, upper bound: 0.0452979
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452979
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0455179, upper bound: 0.0452976
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0459846, upper bound: 0.0462220
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0459139, upper bound: 0.0461419

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0452930
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455147
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0452930
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0452930, upper bound: 0.0455147
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0451849, upper bound: 0.0453801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453522, upper bound: 0.0451668
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453544, upper bound: 0.0450825
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451440, upper bound: 0.0452906
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455144, upper bound: 0.0452194
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451721
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451976
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451075
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451198
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453522, upper bound: 0.0451668
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453544, upper bound: 0.0450825
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450392, upper bound: 0.0452388
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454717, upper bound: 0.0450432
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452413, upper bound: 0.0454924
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452759, upper bound: 0.0454833
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453161
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450274, upper bound: 0.0450419
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451714, upper bound: 0.0451030
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453594, upper bound: 0.0451053
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450392, upper bound: 0.0452386
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450432
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451994, upper bound: 0.0450803
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453426, upper bound: 0.0450718
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 37

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450402, upper bound: 0.0452386
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450432
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451388, upper bound: 0.0454924
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452762, upper bound: 0.0454832
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Candidate
type: RSZ, layer: 3, pos: 19

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0454717
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452388, upper bound: 0.0450392
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450970, upper bound: 0.0453708
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451777, upper bound: 0.0451096
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451760, upper bound: 0.0453785
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451826, upper bound: 0.0451568
time: 0.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0453522, upper bound: 0.0451668
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0453544, upper bound: 0.0450825
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451440, upper bound: 0.0452906
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0455144, upper bound: 0.0452194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451113, upper bound: 0.0451721
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451029, upper bound: 0.0451976
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450915, upper bound: 0.0451075
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451198
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0453522, upper bound: 0.0451668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0453544, upper bound: 0.0450825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450392, upper bound: 0.0452388
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0454717, upper bound: 0.0450432
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0452413, upper bound: 0.0454924
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0452759, upper bound: 0.0454833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450156, upper bound: 0.0453161
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450274, upper bound: 0.0450419
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451714, upper bound: 0.0451030
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0453594, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450392, upper bound: 0.0452386
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450432
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451994, upper bound: 0.0450803
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0453426, upper bound: 0.0450718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450402, upper bound: 0.0452386
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450432
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451388, upper bound: 0.0454924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0452762, upper bound: 0.0454832
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0454717
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0452388, upper bound: 0.0450392
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0450970, upper bound: 0.0453708
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451777, upper bound: 0.0451096
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451760, upper bound: 0.0453785
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.99
Output dim: 0, lower bound: -0.0451826, upper bound: 0.0451568

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450106, upper bound: 0.0451136
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453155, upper bound: 0.0450100
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450838, upper bound: 0.0450710
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453451, upper bound: 0.0450669
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450942, upper bound: 0.0452738
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451190, upper bound: 0.0451358
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454832, upper bound: 0.0452036
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454923, upper bound: 0.0451038
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450783
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450123, upper bound: 0.0450316
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450837, upper bound: 0.0451596
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453431, upper bound: 0.0450734
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450839, upper bound: 0.0450710
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453451, upper bound: 0.0450669
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450098
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450207
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454289, upper bound: 0.0450075
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450862, upper bound: 0.0450998
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450770, upper bound: 0.0451116
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450848, upper bound: 0.0451397
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450752, upper bound: 0.0451652
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449913, upper bound: 0.0453061
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449906, upper bound: 0.0450188
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453141, upper bound: 0.0450156
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450308, upper bound: 0.0452363
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450402, upper bound: 0.0451599
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450308, upper bound: 0.0450432
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450426
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450106, upper bound: 0.0450025
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451367, upper bound: 0.0449904
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451514, upper bound: 0.0450718
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453406, upper bound: 0.0450718
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450123
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450207
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450783, upper bound: 0.0450157
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450529, upper bound: 0.0450203
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450137, upper bound: 0.0454398
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450598, upper bound: 0.0450022
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450849, upper bound: 0.0451402
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450677, upper bound: 0.0451655
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450426, upper bound: 0.0454717
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0450712
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450207, upper bound: 0.0450316
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450098, upper bound: 0.0450316
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0451028
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450802, upper bound: 0.0451182
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451682, upper bound: 0.0451096
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450885, upper bound: 0.0451080
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450964, upper bound: 0.0451179
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451316
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0450869
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451292, upper bound: 0.0450355
time: 0.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450106, upper bound: 0.0451136
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453155, upper bound: 0.0450100
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450838, upper bound: 0.0450710
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453451, upper bound: 0.0450669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450942, upper bound: 0.0452738
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0451190, upper bound: 0.0451358
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0454832, upper bound: 0.0452036
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0454923, upper bound: 0.0451038
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450783
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450123, upper bound: 0.0450316
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450837, upper bound: 0.0451596
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453431, upper bound: 0.0450734
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450839, upper bound: 0.0450710
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453451, upper bound: 0.0450669
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450098
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450207
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0454289, upper bound: 0.0450075
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450862, upper bound: 0.0450998
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450770, upper bound: 0.0451116
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450848, upper bound: 0.0451397
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450752, upper bound: 0.0451652
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0449913, upper bound: 0.0453061
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0449906, upper bound: 0.0450188
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450419, upper bound: 0.0450274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453141, upper bound: 0.0450156
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450308, upper bound: 0.0452363
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450402, upper bound: 0.0451599
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450308, upper bound: 0.0450432
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0454667, upper bound: 0.0450426
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450106, upper bound: 0.0450025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0451367, upper bound: 0.0449904
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0451514, upper bound: 0.0450718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0453406, upper bound: 0.0450718
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450123
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450316, upper bound: 0.0450207
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450783, upper bound: 0.0450157
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450529, upper bound: 0.0450203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450137, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450598, upper bound: 0.0450022
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450849, upper bound: 0.0451402
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450677, upper bound: 0.0451655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450426, upper bound: 0.0454717
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450432, upper bound: 0.0450712
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450207, upper bound: 0.0450316
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450098, upper bound: 0.0450316
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450780, upper bound: 0.0451028
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450802, upper bound: 0.0451182
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0451682, upper bound: 0.0451096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450885, upper bound: 0.0451080
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450964, upper bound: 0.0451179
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450912, upper bound: 0.0451316
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0450393, upper bound: 0.0450869
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -0.0451292, upper bound: 0.0450355

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450188, upper bound: 0.0450100
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453139, upper bound: 0.0450095
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449671
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453090, upper bound: 0.0449773
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450792, upper bound: 0.0452662
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450744, upper bound: 0.0450704
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453721, upper bound: 0.0451974
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454533, upper bound: 0.0450748
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453656, upper bound: 0.0450943
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454546, upper bound: 0.0450732
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450835, upper bound: 0.0450665
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453415, upper bound: 0.0450734
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450801, upper bound: 0.0450565
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450541, upper bound: 0.0450590
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450358, upper bound: 0.0450241
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450532, upper bound: 0.0450034
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454289, upper bound: 0.0450075
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0449902
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0450196
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450291, upper bound: 0.0450059
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450059
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450180, upper bound: 0.0450122
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450166, upper bound: 0.0450200
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454290, upper bound: 0.0450115
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0454397, upper bound: 0.0450135
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450101, upper bound: 0.0449897
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452952, upper bound: 0.0449897
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449949, upper bound: 0.0450059
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449891, upper bound: 0.0450270
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450135, upper bound: 0.0454398
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450127, upper bound: 0.0454398
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450188, upper bound: 0.0450100
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0453139, upper bound: 0.0450095
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449671
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0453090, upper bound: 0.0449773
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450792, upper bound: 0.0452662
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450744, upper bound: 0.0450704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0453721, upper bound: 0.0451974
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0454533, upper bound: 0.0450748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0453656, upper bound: 0.0450943
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0454546, upper bound: 0.0450732
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450835, upper bound: 0.0450665
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0453415, upper bound: 0.0450734
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450801, upper bound: 0.0450565
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450541, upper bound: 0.0450590
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450358, upper bound: 0.0450241
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0453257, upper bound: 0.0450274
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450532, upper bound: 0.0450034
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0454289, upper bound: 0.0450075
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0449902
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0449902, upper bound: 0.0450196
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450291, upper bound: 0.0450059
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450059, upper bound: 0.0450059
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450180, upper bound: 0.0450122
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450166, upper bound: 0.0450200
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0454290, upper bound: 0.0450115
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0454397, upper bound: 0.0450135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450101, upper bound: 0.0449897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0452952, upper bound: 0.0449897
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0449949, upper bound: 0.0450059
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0449891, upper bound: 0.0450270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450135, upper bound: 0.0454398
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 0, lower bound: -0.0450127, upper bound: 0.0454398

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449942, upper bound: 0.0449922
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453047, upper bound: 0.0449839
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450507, upper bound: 0.0449771
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449953, upper bound: 0.0449773
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450714, upper bound: 0.0450541
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450673, upper bound: 0.0450751
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0451383
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450938, upper bound: 0.0450624
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450677, upper bound: 0.0450676
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450957, upper bound: 0.0450637
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450789, upper bound: 0.0450733
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449837
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453925, upper bound: 0.0449832
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450860, upper bound: 0.0450582
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450686, upper bound: 0.0450644
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450477, upper bound: 0.0449999
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450237, upper bound: 0.0450032
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453925, upper bound: 0.0449832
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450507, upper bound: 0.0449886
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450253, upper bound: 0.0449899
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452925, upper bound: 0.0449996
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453915, upper bound: 0.0449831
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0449826
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0450059
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0450270
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449835, upper bound: 0.0453948
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0450009, upper bound: 0.0452965
time: 0.29 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449942, upper bound: 0.0449922
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0453047, upper bound: 0.0449839
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450507, upper bound: 0.0449771
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449953, upper bound: 0.0449773
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450714, upper bound: 0.0450541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450673, upper bound: 0.0450751
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0451383
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0452965, upper bound: 0.0450009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450938, upper bound: 0.0450624
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450677, upper bound: 0.0450676
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450957, upper bound: 0.0450637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450789, upper bound: 0.0450733
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449671, upper bound: 0.0449837
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0453925, upper bound: 0.0449832
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450860, upper bound: 0.0450582
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450686, upper bound: 0.0450644
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450477, upper bound: 0.0449999
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450237, upper bound: 0.0450032
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0453948, upper bound: 0.0449835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0453925, upper bound: 0.0449832
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450507, upper bound: 0.0449886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450253, upper bound: 0.0449899
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0452925, upper bound: 0.0449996
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0453915, upper bound: 0.0449831
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449996, upper bound: 0.0449826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0449826
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449910, upper bound: 0.0450059
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449826, upper bound: 0.0450270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0449835, upper bound: 0.0453948
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -0.0450009, upper bound: 0.0452965

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449957, upper bound: 0.0449710
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449771, upper bound: 0.0449736
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449762
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450034, upper bound: 0.0449751
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449739, upper bound: 0.0449777
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 37
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 37

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449948, upper bound: 0.0449762
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449816, upper bound: 0.0449805
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449948, upper bound: 0.0449762
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449777, upper bound: 0.0449739
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449751, upper bound: 0.0450034
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 5
type: RSZ, layer: 5, pos: 21
type: RSZ, layer: 5, pos: 8
type: RSZ, layer: 5, pos: 9
type: RSZ, layer: 5, pos: 44
type: RSZ, layer: 5, pos: 25
type: RSZ, layer: 5, pos: 40
type: RSZ, layer: 5, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 5, pos: 21

### Candidate
type: RSZ, layer: 5, pos: 8

### Candidate
type: RSZ, layer: 5, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 5, pos: 25

### Candidate
type: RSZ, layer: 5, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449939
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449727, upper bound: 0.0450179
time: 0.30 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449957, upper bound: 0.0449710
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449771, upper bound: 0.0449736
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0450179, upper bound: 0.0449728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449938, upper bound: 0.0449779
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449762
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0450034, upper bound: 0.0449751
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449739, upper bound: 0.0449777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449948, upper bound: 0.0449762
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0450072, upper bound: 0.0449761
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449816, upper bound: 0.0449805
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449948, upper bound: 0.0449762
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449670, upper bound: 0.0449782
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449777, upper bound: 0.0449739
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449751, upper bound: 0.0450034
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449779, upper bound: 0.0449939
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.58
Output dim: 0, lower bound: -0.0449727, upper bound: 0.0450179
Binary search (step 8): status=Status.VERIFIED, low=0.1814631, high=0.1818182, mid=0.1814631, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 9) starts
Candidate diff: 0.1816406


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -0.0463383, upper bound: 0.0463383

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.71 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462663
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462664, upper bound: 0.0461733
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.73 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455825
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455825
time: 0.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 0, lower bound: -0.0461733, upper bound: 0.0462663
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 0, lower bound: -0.0462664, upper bound: 0.0461733
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455825
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 0, lower bound: -0.0455822, upper bound: 0.0455825

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462220, upper bound: 0.0459846
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0458379, upper bound: 0.0461293
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0453801
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
time: 0.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0462220, upper bound: 0.0459846
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0458379, upper bound: 0.0461293
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0455767
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0453614, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0451850
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455147
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0452930
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 37
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 37

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0451053
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455147
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453801
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455147
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0453801, upper bound: 0.0451850
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0452927, upper bound: 0.0455147
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0455146, upper bound: 0.0452930
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453614
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0451053
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0455147
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0455147, upper bound: 0.0452927
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.17
Output dim: 0, lower bound: -0.0451053, upper bound: 0.0453801
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0452976
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.17
Output dim: 0, lower bound: -0.0452976, upper bound: 0.0455179
Binary search (step 9): status=Status.UNKNOWN, low=0.1814631, high=0.1816406, mid=0.1816406, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.18146307358983904
execution time: 1150.54 seconds
