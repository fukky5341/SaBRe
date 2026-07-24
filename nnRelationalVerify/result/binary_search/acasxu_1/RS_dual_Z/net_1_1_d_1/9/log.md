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
execution time: IAR + LP analysis = 2.10 + 1.25 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.3982887, upper bound: 3.3982887


# Binary Search by BASE starts (time budget: 1196.65 seconds, max iter: 100)

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
Binary search time: 65.70 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1130.95 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3966881, upper bound: 3.3963644
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3963644, upper bound: 3.3966881
time: 0.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 0, lower bound: -3.3966881, upper bound: 3.3963644
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.05
Output dim: 0, lower bound: -3.3963644, upper bound: 3.3966881

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.91
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871

## BFS RS instance: RS_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.42 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.07
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804

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
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
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
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.42 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.43
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982802470538505, 3.398280247053849]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3965735, upper bound: 3.3963644
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3963644, upper bound: 3.3965735
time: 0.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 0, lower bound: -3.3965735, upper bound: 3.3963644
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 0, lower bound: -3.3963644, upper bound: 3.3965735

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.41 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
time: 0.44 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3.3359871, upper bound: 3.3359871

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
time: 0.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 0, lower bound: -3.3357804, upper bound: 3.3357804

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
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
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.43 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

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
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3119625, upper bound: 3.3119625
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.49 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=3.5238394737243652
rel_dist={0: [-3.3982329481346, 3.3982329481346003]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3964854, upper bound: 3.3963644
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3963644, upper bound: 3.3964854
time: 0.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 0, lower bound: -3.3964854, upper bound: 3.3963644
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 0, lower bound: -3.3963644, upper bound: 3.3964854

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
time: 0.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 0, lower bound: -3.3359810, upper bound: 3.3359810

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
time: 0.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 0, lower bound: -3.3357744, upper bound: 3.3357744

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
time: 0.41 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -3.3356458, upper bound: 3.3356458

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
time: 0.43 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

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
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.2466472, 2.2771921, -1.2466472, 2.2771921, -3.5238395, 3.5238395
1: -1.9637374, 3.1950235, -1.9637374, 3.1950235, -5.1587601, 5.1587596
2: -1.3604455, 3.2710245, -1.3604455, 3.2710245, -4.6314697, 4.6314697
3: -3.4595599, 4.0564485, -3.4595599, 4.0564485, -7.5160079, 7.5160084
4: -2.1785955, 4.2066536, -2.1785955, 4.2066536, -6.3852487, 6.3852487

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
time: 0.44 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346434
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346709, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346709
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346431, upper bound: 3.3346431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -3.3346434, upper bound: 3.3346431
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=3.5238394737243652
rel_dist={0: [-3.398203369636585, 3.398203369636585]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1131.30 seconds
