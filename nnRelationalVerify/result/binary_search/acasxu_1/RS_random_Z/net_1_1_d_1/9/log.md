## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 3.05840151


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395)
1: (-1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596)
2: (-1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697)
3: (-3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084)
4: (-2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487)

## BASE Result
execution time: IAR + LP analysis = 2.12 + 1.26 = 3.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982887, upper bound: 3.3982887


# Binary Search by BASE starts (time budget: 1196.62 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982802470538505, 3.398280247053849]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982329481346, 3.3982329481346003]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=3.5238394737243652
rel_dist={0: [-3.398203369636585, 3.398203369636585]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=3.5238394737243652
rel_dist={0: [-3.39818007524518, 3.398180075245179]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=3.5238394737243652
rel_dist={0: [-3.3981656742023487, 3.3981656742023487]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=3.5238394737243652
rel_dist={0: [-3.3981560004933664, 3.3981560004933664]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=3.5238394737243652
rel_dist={0: [-3.398151163638105, 3.398151163638106]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=3.5238394737243652
rel_dist={0: [-3.398148745209495, 3.398148745209495]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=3.5238394737243652
rel_dist={0: [-3.398147442234788, 3.398147442234788]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=3.5238394737243652
rel_dist={0: [-3.3981467631830444, 3.3981467631830444]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=3.5238394737243652
rel_dist={0: [-3.3981464236572503, 3.3981464236572503]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=3.5238394737243652
rel_dist={0: [-3.398146253894503, 3.3981462538945024]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=3.5238394737243652
rel_dist={0: [-3.3981461690134305, 3.3981461690134305]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=3.5238394737243652
rel_dist={0: [-3.398146126573492, 3.398146126573492]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=3.5238394737243652
rel_dist={0: [-3.3981461053547006, 3.3981461053547006]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=3.5238394737243652
rel_dist={0: [-3.398146094747597, 3.398146094747597]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=3.5238394737243652
rel_dist={0: [-3.3981460894483844, 3.3981460894483844]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=3.5238394737243652
rel_dist={0: [-3.3981460896207114, 3.3981460868065954]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=3.5238394737243652
rel_dist={0: [-3.3981461239315784, 3.3981460938166226]}

## Binary Search Result
Binary search time: 65.81 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1130.80 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3982802, upper bound: 3.3982802
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3982802, upper bound: 3.3982802
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -3.3982802, upper bound: 3.3982802
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -3.3982802, upper bound: 3.3982802

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3619032, upper bound: 3.3619034
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3619032, upper bound: 3.3619034
time: 0.42 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3946814, upper bound: 3.3946796
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3946796, upper bound: 3.3946814
time: 0.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3619032, upper bound: 3.3619034
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3619032, upper bound: 3.3619034
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3946814, upper bound: 3.3946796
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -3.3946796, upper bound: 3.3946814

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3452929, upper bound: 3.3454148
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3453558, upper bound: 3.3453012
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3611346, upper bound: 3.3613554
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3613572, upper bound: 3.3611353
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3862930, upper bound: 3.3862930
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3862930, upper bound: 3.3862930
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3452929, upper bound: 3.3454148
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3453558, upper bound: 3.3453012
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3611346, upper bound: 3.3613554
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3613572, upper bound: 3.3611353
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3862930, upper bound: 3.3862930
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3862930, upper bound: 3.3862930
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3452929, upper bound: 3.3454148
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3452929, upper bound: 3.3452929
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442785
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442618, upper bound: 3.3442027
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3421846, upper bound: 3.3421728
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3421846, upper bound: 3.3421728
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3452929, upper bound: 3.3454148
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3452929, upper bound: 3.3452929
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442785
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3442618, upper bound: 3.3442027
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3421846, upper bound: 3.3421728
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3421846, upper bound: 3.3421728
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -3.3724576, upper bound: 3.3724576

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3312699, upper bound: 3.3315499
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3312699, upper bound: 3.3315499
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442025
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442025
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442785
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442027
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442025
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3442618, upper bound: 3.3442027
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3647269
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3647269
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3312699, upper bound: 3.3315499
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3312699, upper bound: 3.3315499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442025
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442025
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442785
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442027
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3442025, upper bound: 3.3442025
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3442618, upper bound: 3.3442027
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.0213540, upper bound: 3.0213540
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3135254, upper bound: 3.3135254
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3122901, upper bound: 3.3122901
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3647269
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3647269
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -3.3720194, upper bound: 3.3720194

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3307020
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3307020, upper bound: 3.3304830
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3307020, upper bound: 3.3304830
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.50 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3307020
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3438732, upper bound: 3.3438732
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3307020, upper bound: 3.3304830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3307020, upper bound: 3.3304830
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3133915, upper bound: 3.3133915
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3133175, upper bound: 3.3133175
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3590427, upper bound: 3.3590427
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3203361, upper bound: 3.3203361
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3203361, upper bound: 3.3203361
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304465, upper bound: 3.3302015
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315562, upper bound: 3.3315555
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315562, upper bound: 3.3315555
time: 0.42 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3203361, upper bound: 3.3203361
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3203361, upper bound: 3.3203361
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3304465, upper bound: 3.3302015
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3131836, upper bound: 3.3131836
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3120897, upper bound: 3.3120897
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315562, upper bound: 3.3315555
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.79
Output dim: 0, lower bound: -3.3315562, upper bound: 3.3315555

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.36 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982802470538505, 3.398280247053849]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3619019, upper bound: 3.3619019
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3619019, upper bound: 3.3619019
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -3.3619019, upper bound: 3.3619019
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -3.3619019, upper bound: 3.3619019

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3505898, upper bound: 3.3505898
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3505898, upper bound: 3.3505898
time: 0.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0222082, upper bound: 3.0222082
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3.0222082, upper bound: 3.0222082
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 0, lower bound: -3.3505898, upper bound: 3.3505898
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 0, lower bound: -3.3505898, upper bound: 3.3505898
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 0, lower bound: -3.0222082, upper bound: 3.0222082
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.73
Output dim: 0, lower bound: -3.0222082, upper bound: 3.0222082

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3319424, upper bound: 3.3320385
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3320385, upper bound: 3.3319424
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338161, upper bound: 3.3339016
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338635, upper bound: 3.3338954
time: 0.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -3.3319424, upper bound: 3.3320385
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -3.3320385, upper bound: 3.3319424
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -3.3338161, upper bound: 3.3339016
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -3.3338635, upper bound: 3.3338954

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3318883, upper bound: 3.3320385
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3318883, upper bound: 3.3319904
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120220, upper bound: 3.3120220
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120291, upper bound: 3.3120220
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120220, upper bound: 3.3120291
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3120220, upper bound: 3.3120220
time: 0.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3318883, upper bound: 3.3320385
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3318883, upper bound: 3.3319904
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3120220, upper bound: 3.3120220
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3120291, upper bound: 3.3120220
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3120220, upper bound: 3.3120291
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 0, lower bound: -3.3120220, upper bound: 3.3120220

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3305790, upper bound: 3.3307822
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3305790, upper bound: 3.3305803
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3137138, upper bound: 3.3137967
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3137138, upper bound: 3.3137967
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118796
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118773
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3305790, upper bound: 3.3307822
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3305790, upper bound: 3.3305803
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3137138, upper bound: 3.3137967
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3137138, upper bound: 3.3137967
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118796
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118773
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100587
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3304830, upper bound: 3.3304830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3105845, upper bound: 3.3105845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100587
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.93
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100581, upper bound: 3.3100580
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100587
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.50 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3121629, upper bound: 3.3121629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3302015, upper bound: 3.3302015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100581, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100587
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.25
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 1.89 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 1.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2882699, upper bound: 3.2882699
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2882699, upper bound: 3.2882699
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 1.93 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 2.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 2.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001732, upper bound: 3.3001732
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001732, upper bound: 3.3001732
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 2.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 1.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 1.94 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 32

Time for candidate selection: 1.94 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 34

Time for candidate selection: 1.99 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 1.96 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001732, upper bound: 3.3001732
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3001732, upper bound: 3.3001732
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 2.01 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2882699, upper bound: 3.2882699
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2882699, upper bound: 3.2882699
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 1.94 seconds

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.97 seconds

### Candidate
type: RSZ, layer: 3, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2882699, upper bound: 3.2882699
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.2882699, upper bound: 3.2882699
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 2.03 seconds

### Candidate
type: RSZ, layer: 3, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3095477, upper bound: 3.3095477
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.46 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982329481346, 3.3982329481346003]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3911078, upper bound: 3.3911102
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3911102, upper bound: 3.3911078
time: 0.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -3.3911078, upper bound: 3.3911102
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -3.3911102, upper bound: 3.3911078

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3885653, upper bound: 3.3885730
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3885702, upper bound: 3.3885730
time: 0.43 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3832018, upper bound: 3.3832061
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3832310, upper bound: 3.3832009
time: 0.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -3.3885653, upper bound: 3.3885730
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -3.3885702, upper bound: 3.3885730
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -3.3832018, upper bound: 3.3832061
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.00
Output dim: 0, lower bound: -3.3832310, upper bound: 3.3832009

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3863707, upper bound: 3.3864144
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3863707, upper bound: 3.3864088
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3385943, upper bound: 3.3385411
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3385943, upper bound: 3.3385411
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3787642, upper bound: 3.3786507
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3787642, upper bound: 3.3786507
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3564561, upper bound: 3.3564484
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3564561, upper bound: 3.3564484
time: 0.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3863707, upper bound: 3.3864144
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3863707, upper bound: 3.3864088
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3385943, upper bound: 3.3385411
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3385943, upper bound: 3.3385411
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3787642, upper bound: 3.3786507
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3787642, upper bound: 3.3786507
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3564561, upper bound: 3.3564484
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 0, lower bound: -3.3564561, upper bound: 3.3564484

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650301, upper bound: 3.3650042
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650301, upper bound: 3.3650042
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338920, upper bound: 3.3338216
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3338161, upper bound: 3.3338161
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3381530, upper bound: 3.3381530
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3382237, upper bound: 3.3381530
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3167385, upper bound: 3.3167385
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3167385, upper bound: 3.3167385
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3167413, upper bound: 3.3167385
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3167413, upper bound: 3.3167385
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3149185, upper bound: 3.3149178
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3149185, upper bound: 3.3149178
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3149178, upper bound: 3.3149178
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3149178, upper bound: 3.3149178
time: 0.41 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3650301, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3650301, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3338920, upper bound: 3.3338216
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3338161, upper bound: 3.3338161
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3381530, upper bound: 3.3381530
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3382237, upper bound: 3.3381530
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3167385, upper bound: 3.3167385
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3167385, upper bound: 3.3167385
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3167413, upper bound: 3.3167385
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3167413, upper bound: 3.3167385
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3149185, upper bound: 3.3149178
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3149185, upper bound: 3.3149178
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3149178, upper bound: 3.3149178
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 0, lower bound: -3.3149178, upper bound: 3.3149178

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650301, upper bound: 3.3650042
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3334311, upper bound: 3.3334367
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3335240, upper bound: 3.3334311
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3329146, upper bound: 3.3329146
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3329146, upper bound: 3.3329146
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3345263, upper bound: 3.3345263
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3345263, upper bound: 3.3345263
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3334311, upper bound: 3.3334311
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3334311, upper bound: 3.3334311
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3165922, upper bound: 3.3165922
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3165922, upper bound: 3.3165922
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3110021, upper bound: 3.3110021
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3110021, upper bound: 3.3110021
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102182, upper bound: 3.3102175
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102179, upper bound: 3.3102175
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147605, upper bound: 3.3147598
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147605, upper bound: 3.3147598
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147598, upper bound: 3.3147598
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147598, upper bound: 3.3147598
time: 0.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3650301, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3650042, upper bound: 3.3650042
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3334311, upper bound: 3.3334367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3335240, upper bound: 3.3334311
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3329146, upper bound: 3.3329146
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3329146, upper bound: 3.3329146
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3345263, upper bound: 3.3345263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3345263, upper bound: 3.3345263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3334311, upper bound: 3.3334311
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3334311, upper bound: 3.3334311
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3165922, upper bound: 3.3165922
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3165922, upper bound: 3.3165922
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3110021, upper bound: 3.3110021
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3110021, upper bound: 3.3110021
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3157512, upper bound: 3.3157512
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3102182, upper bound: 3.3102175
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3102179, upper bound: 3.3102175
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3147605, upper bound: 3.3147598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3147605, upper bound: 3.3147598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3147598, upper bound: 3.3147598
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.25
Output dim: 0, lower bound: -3.3147598, upper bound: 3.3147598

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3647269, upper bound: 3.3646969
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533973, upper bound: 3.3533973
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533973, upper bound: 3.3533973
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118732
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118732
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147605, upper bound: 3.3147598
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147598, upper bound: 3.3147598
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100587, upper bound: 3.3100580
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100584, upper bound: 3.3100580
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3647269, upper bound: 3.3646969
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3536465, upper bound: 3.3536465
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3317906, upper bound: 3.3317911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3646969, upper bound: 3.3646969
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3533973, upper bound: 3.3533973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3533973, upper bound: 3.3533973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3102175, upper bound: 3.3102175
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118732
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3118732, upper bound: 3.3118732
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3326684, upper bound: 3.3326684
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3302684, upper bound: 3.3302684
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3107995, upper bound: 3.3107995
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3108041, upper bound: 3.3108041
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3075554, upper bound: 3.3075554
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3155388, upper bound: 3.3155388
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147605, upper bound: 3.3147598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147598, upper bound: 3.3147598
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3100587, upper bound: 3.3100580
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3100584, upper bound: 3.3100580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3147022, upper bound: 3.3147022
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 0, lower bound: -3.3144898, upper bound: 3.3144898

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315562
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3533895, upper bound: 3.3533895
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3097601, upper bound: 3.3097601
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3100580, upper bound: 3.3100580
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3073494, upper bound: 3.3073494
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3315555, upper bound: 3.3315555
time: 0.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=3.5238394737243652
rel_dist={0: [-3.398203369636585, 3.398203369636585]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1132.54 seconds
