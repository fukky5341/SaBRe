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
execution time: IAR + LP analysis = 1.57 + 0.84 = 2.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0465726, upper bound: 0.0465726


# Binary Search by BASE starts (time budget: 1197.59 seconds, max iter: 100)

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
Binary search time: 47.09 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1150.49 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462642, upper bound: 0.0462653
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462643
time: 0.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462642, upper bound: 0.0462653
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462643

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
time: 0.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
Binary search (step 0): status=Status.VERIFIED, low=0.0909091, high=0.1818182, mid=0.0909091, abs_max=0.05155529826879501
rel_dist={0: [-0.04657254158633466, 0.04657254158633465]}

## Binary search (step 1) starts
Candidate diff: 0.1363636


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
time: 0.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553
1: -0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822
2: -0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808
3: -0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506
4: -0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449321
Binary search (step 1): status=Status.VERIFIED, low=0.1363636, high=0.1818182, mid=0.1363636, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 2) starts
Candidate diff: 0.1590909


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 2): status=Status.VERIFIED, low=0.1590909, high=0.1818182, mid=0.1590909, abs_max=0.05155529826879501
rel_dist={0: [-0.0465725463803796, 0.04657255964042008]}

## Binary search (step 3) starts
Candidate diff: 0.1704546


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.79 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
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

Time for candidate selection: 0.13 seconds

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
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.69 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.69
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

Time for candidate selection: 0.15 seconds

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
- Time for RS candidates: 0.71 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.71
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
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 6): status=Status.VERIFIED, low=0.1803977, high=0.1818182, mid=0.1803977, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 7) starts
Candidate diff: 0.1811080


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 7): status=Status.VERIFIED, low=0.1811080, high=0.1818182, mid=0.1811080, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 8) starts
Candidate diff: 0.1814631


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

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
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 8): status=Status.VERIFIED, low=0.1814631, high=0.1818182, mid=0.1814631, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 9) starts
Candidate diff: 0.1816406


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 9): status=Status.VERIFIED, low=0.1816406, high=0.1818182, mid=0.1816406, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 10) starts
Candidate diff: 0.1817294


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 10): status=Status.VERIFIED, low=0.1817294, high=0.1818182, mid=0.1817294, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 11) starts
Candidate diff: 0.1817738


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

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
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 11): status=Status.VERIFIED, low=0.1817738, high=0.1818182, mid=0.1817738, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 12) starts
Candidate diff: 0.1817960


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

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
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 12): status=Status.VERIFIED, low=0.1817960, high=0.1818182, mid=0.1817960, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 13) starts
Candidate diff: 0.1818071


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.71 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 13): status=Status.VERIFIED, low=0.1818071, high=0.1818182, mid=0.1818071, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 14) starts
Candidate diff: 0.1818126


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

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
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.69 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 14): status=Status.VERIFIED, low=0.1818126, high=0.1818182, mid=0.1818126, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 15) starts
Candidate diff: 0.1818154


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

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
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.69 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.69
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 15): status=Status.VERIFIED, low=0.1818154, high=0.1818182, mid=0.1818154, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 16) starts
Candidate diff: 0.1818168


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

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
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.70 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 16): status=Status.VERIFIED, low=0.1818168, high=0.1818182, mid=0.1818168, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 17) starts
Candidate diff: 0.1818175


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

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
time: 0.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
Binary search (step 17): status=Status.VERIFIED, low=0.1818175, high=0.1818182, mid=0.1818175, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1818174936554442
execution time: 70.56 seconds
