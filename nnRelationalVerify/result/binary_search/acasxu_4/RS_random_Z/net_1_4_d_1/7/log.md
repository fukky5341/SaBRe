## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 398.85261092052


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315)
1: (-197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482)
2: (-197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371)
3: (-234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619)
4: (-201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945)

## BASE Result
execution time: IAR + LP analysis = 2.45 + 2.43 = 4.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9390144, upper bound: 398.9390144


# Binary Search by BASE starts (time budget: 1195.12 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=482.57733154296875
rel_dist={0: [-398.93901443925324, 398.93901443925324]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=482.57733154296875
rel_dist={0: [-398.9373570313671, 398.9373570313671]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=482.57733154296875
rel_dist={0: [-398.93352538884415, 398.93352538884415]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=482.57733154296875
rel_dist={0: [-398.9304526193936, 398.9304526193936]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=482.57733154296875
rel_dist={0: [-398.9286294160353, 398.9286294160353]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=482.57733154296875
rel_dist={0: [-398.9271246178754, 398.9271246178754]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=482.57733154296875
rel_dist={0: [-398.92607928857916, 398.9260792885791]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=482.57733154296875
rel_dist={0: [-398.9255252454213, 398.92552524542134]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=482.57733154296875
rel_dist={0: [-398.92524179851074, 398.9252417985108]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=482.57733154296875
rel_dist={0: [-398.92509988427423, 398.92509988427435]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=482.57733154296875
rel_dist={0: [-398.9250267544338, 398.92502675443393]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=482.57733154296875
rel_dist={0: [-398.9249900282473, 398.9249900282473]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=482.57733154296875
rel_dist={0: [-398.9249713707402, 398.9249713707402]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=482.57733154296875
rel_dist={0: [-398.9249620121283, 398.9249620142207]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=482.57733154296875
rel_dist={0: [-398.9249573327561, 398.9249573331433]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=482.57733154296875
rel_dist={0: [-398.9249549926937, 398.92495499269376]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=482.57733154296875
rel_dist={0: [-398.9249538490367, 398.92495382392326]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=482.57733154296875
rel_dist={0: [-398.9249532635105, 398.924953264239]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=482.57733154296875
rel_dist={0: [-398.92495298950195, 398.92495302375687]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=482.57733154296875
rel_dist={0: [-398.9249528754319, 398.92495289837075]}

## Binary Search Result
Binary search time: 99.81 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1095.31 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9390041, upper bound: 398.9390144
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9390144, upper bound: 398.9390041
time: 0.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -398.9390041, upper bound: 398.9390144
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -398.9390144, upper bound: 398.9390041

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8699195, upper bound: 398.8699195
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8699195, upper bound: 398.8699195
time: 1.24 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9389134, upper bound: 398.9369933
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9372745, upper bound: 398.9389142
time: 1.06 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.40
Output dim: 0, lower bound: -398.8699195, upper bound: 398.8699195
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.40
Output dim: 0, lower bound: -398.8699195, upper bound: 398.8699195
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.40
Output dim: 0, lower bound: -398.9389134, upper bound: 398.9369933
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.40
Output dim: 0, lower bound: -398.9372745, upper bound: 398.9389142

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9382933, upper bound: 398.9314051
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9358481, upper bound: 398.9356967
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9328947, upper bound: 398.9389085
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9372745, upper bound: 398.9332125
time: 1.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.8698433, upper bound: 398.8698433
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.9382933, upper bound: 398.9314051
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.9358481, upper bound: 398.9356967
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.9328947, upper bound: 398.9389085
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 0, lower bound: -398.9372745, upper bound: 398.9332125

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8664926, upper bound: 398.8664926
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8664926, upper bound: 398.8664926
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8689393, upper bound: 398.8689393
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8689393, upper bound: 398.8689393
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9304058, upper bound: 398.9303109
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9374303, upper bound: 398.9302279
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310745, upper bound: 398.9332425
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9333036, upper bound: 398.9295722
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9322518, upper bound: 398.9386010
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9325242, upper bound: 398.9310710
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9316364, upper bound: 398.9271289
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9253229, upper bound: 398.9259868
time: 1.24 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8664926, upper bound: 398.8664926
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8664926, upper bound: 398.8664926
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8689393, upper bound: 398.8689393
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8689393, upper bound: 398.8689393
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9304058, upper bound: 398.9303109
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9374303, upper bound: 398.9302279
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9310745, upper bound: 398.9332425
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9333036, upper bound: 398.9295722
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9322518, upper bound: 398.9386010
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9325242, upper bound: 398.9310710
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9316364, upper bound: 398.9271289
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -398.9253229, upper bound: 398.9259868

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8644562, upper bound: 398.8644562
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8644562, upper bound: 398.8644562
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8505246, upper bound: 398.8505246
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8505246, upper bound: 398.8505246
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8689389, upper bound: 398.8689389
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8689389, upper bound: 398.8689389
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655110, upper bound: 398.8655088
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655093, upper bound: 398.8655063
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682282, upper bound: 398.8682282
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682282, upper bound: 398.8682282
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9301891, upper bound: 398.9282998
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9328328, upper bound: 398.9282967
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310540, upper bound: 398.9332425
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310502, upper bound: 398.9323487
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310922, upper bound: 398.9295722
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9332732, upper bound: 398.9295230
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9248382, upper bound: 398.9316970
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9263006, upper bound: 398.9317035
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9321698, upper bound: 398.9307697
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9312873, upper bound: 398.9307439
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9313571, upper bound: 398.9269929
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9315565, upper bound: 398.9264927
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9321849, upper bound: 398.9253175
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9320318, upper bound: 398.9259868
time: 1.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8644562, upper bound: 398.8644562
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8644562, upper bound: 398.8644562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8505246, upper bound: 398.8505246
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8505246, upper bound: 398.8505246
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8689389, upper bound: 398.8689389
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8689389, upper bound: 398.8689389
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8655110, upper bound: 398.8655088
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8655093, upper bound: 398.8655063
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8682282, upper bound: 398.8682282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.8682282, upper bound: 398.8682282
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9301891, upper bound: 398.9282998
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9328328, upper bound: 398.9282967
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9310540, upper bound: 398.9332425
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9310502, upper bound: 398.9323487
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9310922, upper bound: 398.9295722
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9332732, upper bound: 398.9295230
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9248382, upper bound: 398.9316970
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9263006, upper bound: 398.9317035
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9321698, upper bound: 398.9307697
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9312873, upper bound: 398.9307439
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9313571, upper bound: 398.9269929
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9315565, upper bound: 398.9264927
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9321849, upper bound: 398.9253175
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.02
Output dim: 0, lower bound: -398.9320318, upper bound: 398.9259868

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8597481, upper bound: 398.8599578
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8598708, upper bound: 398.8597481
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8615328, upper bound: 398.8615328
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8615880, upper bound: 398.8615328
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8658888, upper bound: 398.8658888
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8658888, upper bound: 398.8658888
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651472, upper bound: 398.8651472
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651472, upper bound: 398.8651472
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655105, upper bound: 398.8655081
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655110, upper bound: 398.8655088
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650025, upper bound: 398.8650025
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650025, upper bound: 398.8650025
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8679471, upper bound: 398.8679472
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8679472, upper bound: 398.8679470
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8659475, upper bound: 398.8659475
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8659475, upper bound: 398.8659475
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9282482, upper bound: 398.9259686
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9279311, upper bound: 398.9260382
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9277100, upper bound: 398.9280631
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9325719, upper bound: 398.9280648
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9297904, upper bound: 398.9332425
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310540, upper bound: 398.9321715
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9300503, upper bound: 398.9323487
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310502, upper bound: 398.9316112
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9290236, upper bound: 398.9292279
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9306961, upper bound: 398.9291494
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9332619, upper bound: 398.9294862
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9332732, upper bound: 398.9295230
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9245209, upper bound: 398.9314466
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9245254, upper bound: 398.9271821
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9254868, upper bound: 398.9282345
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9248822, upper bound: 398.9268427
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9321693, upper bound: 398.9307697
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310511, upper bound: 398.9307439
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9288856, upper bound: 398.9288856
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9299900, upper bound: 398.9288856
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9308757, upper bound: 398.9267463
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9252871, upper bound: 398.9264568
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8644557, upper bound: 398.8644557
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8644557, upper bound: 398.8644557
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9233659, upper bound: 398.9230961
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9278180, upper bound: 398.9230855
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9148048, upper bound: 398.9125049
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9160445, upper bound: 398.9112054
time: 1.00 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8597481, upper bound: 398.8599578
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8598708, upper bound: 398.8597481
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8615328, upper bound: 398.8615328
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8615880, upper bound: 398.8615328
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8658888, upper bound: 398.8658888
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8658888, upper bound: 398.8658888
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8651472, upper bound: 398.8651472
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8651472, upper bound: 398.8651472
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8655105, upper bound: 398.8655081
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8655110, upper bound: 398.8655088
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8650025, upper bound: 398.8650025
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8650025, upper bound: 398.8650025
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8682948, upper bound: 398.8682948
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8679471, upper bound: 398.8679472
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8679472, upper bound: 398.8679470
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8659475, upper bound: 398.8659475
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8659475, upper bound: 398.8659475
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8674886, upper bound: 398.8674886
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9282482, upper bound: 398.9259686
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9279311, upper bound: 398.9260382
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9277100, upper bound: 398.9280631
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9325719, upper bound: 398.9280648
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9297904, upper bound: 398.9332425
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9310540, upper bound: 398.9321715
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9300503, upper bound: 398.9323487
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9310502, upper bound: 398.9316112
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9290236, upper bound: 398.9292279
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9306961, upper bound: 398.9291494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9332619, upper bound: 398.9294862
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9332732, upper bound: 398.9295230
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9245209, upper bound: 398.9314466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9245254, upper bound: 398.9271821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9254868, upper bound: 398.9282345
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9248822, upper bound: 398.9268427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9321693, upper bound: 398.9307697
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9310511, upper bound: 398.9307439
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9288856, upper bound: 398.9288856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9299900, upper bound: 398.9288856
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9308757, upper bound: 398.9267463
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9252871, upper bound: 398.9264568
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8644557, upper bound: 398.8644557
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.8644557, upper bound: 398.8644557
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9233659, upper bound: 398.9230961
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9278180, upper bound: 398.9230855
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9148048, upper bound: 398.9125049
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 0, lower bound: -398.9160445, upper bound: 398.9112054

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8597481, upper bound: 398.8599578
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8597481, upper bound: 398.8597481
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8567809, upper bound: 398.8567809
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8569092, upper bound: 398.8567809
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8607696, upper bound: 398.8607696
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8607696, upper bound: 398.8607696
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8580948, upper bound: 398.8580771
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8581281, upper bound: 398.8580771
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8658888, upper bound: 398.8658888
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8658888, upper bound: 398.8658888
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650035, upper bound: 398.8650035
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650035, upper bound: 398.8650035
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8630422, upper bound: 398.8630422
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8630670, upper bound: 398.8630422
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660682, upper bound: 398.8660682
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8598902, upper bound: 398.8599245
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8601033, upper bound: 398.8598902
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8669229, upper bound: 398.8669229
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8624652, upper bound: 398.8625039
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8624652, upper bound: 398.8624652
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8437241, upper bound: 398.8437241
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8437241, upper bound: 398.8437241
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8437241, upper bound: 398.8437241
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8437241, upper bound: 398.8437241
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643057, upper bound: 398.8643057
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655729, upper bound: 398.8655729
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660893, upper bound: 398.8660893
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660893, upper bound: 398.8660893
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673138, upper bound: 398.8673138
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673138, upper bound: 398.8673138
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8625270, upper bound: 398.8625269
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8625270, upper bound: 398.8625269
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646226, upper bound: 398.8646226
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646226, upper bound: 398.8646226
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=482.57733154296875
rel_dist={0: [-398.93901443925324, 398.93901443925324]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9316210, upper bound: 398.9316283
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9316283, upper bound: 398.9316210
time: 1.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 0, lower bound: -398.9316210, upper bound: 398.9316283
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 0, lower bound: -398.9316283, upper bound: 398.9316210

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9316210, upper bound: 398.9316006
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9315929, upper bound: 398.9316283
time: 1.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9278350, upper bound: 398.9276079
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9279470, upper bound: 398.9265485
time: 1.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -398.9316210, upper bound: 398.9316006
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -398.9315929, upper bound: 398.9316283
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -398.9278350, upper bound: 398.9276079
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -398.9279470, upper bound: 398.9265485

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9303999, upper bound: 398.9309440
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9309571, upper bound: 398.9304071
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9276197, upper bound: 398.9301062
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9300670, upper bound: 398.9285903
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9278344, upper bound: 398.9275610
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9278350, upper bound: 398.9274956
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9242168, upper bound: 398.9241971
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9241783, upper bound: 398.9241971
time: 1.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9303999, upper bound: 398.9309440
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9309571, upper bound: 398.9304071
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9276197, upper bound: 398.9301062
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9300670, upper bound: 398.9285903
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9278344, upper bound: 398.9275610
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9278350, upper bound: 398.9274956
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9242168, upper bound: 398.9241971
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 0, lower bound: -398.9241783, upper bound: 398.9241971

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9160408, upper bound: 398.9168549
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9144381, upper bound: 398.9168468
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9239460, upper bound: 398.9304071
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9309571, upper bound: 398.9276387
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795940, upper bound: 398.8816156
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8795211, upper bound: 398.8817818
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9299871, upper bound: 398.9284768
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9253083, upper bound: 398.9284445
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9277530, upper bound: 398.9275610
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9278344, upper bound: 398.9271592
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9241991, upper bound: 398.9267954
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9274147, upper bound: 398.9257963
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9237958, upper bound: 398.9238303
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9238715, upper bound: 398.9237958
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9235234, upper bound: 398.9237854
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9240585, upper bound: 398.9233716
time: 1.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9160408, upper bound: 398.9168549
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9144381, upper bound: 398.9168468
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9239460, upper bound: 398.9304071
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9309571, upper bound: 398.9276387
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.8795940, upper bound: 398.8816156
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.8795211, upper bound: 398.8817818
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9299871, upper bound: 398.9284768
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9253083, upper bound: 398.9284445
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9277530, upper bound: 398.9275610
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9278344, upper bound: 398.9271592
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9241991, upper bound: 398.9267954
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9274147, upper bound: 398.9257963
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9237958, upper bound: 398.9238303
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9238715, upper bound: 398.9237958
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9235234, upper bound: 398.9237854
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.21
Output dim: 0, lower bound: -398.9240585, upper bound: 398.9233716

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9153829, upper bound: 398.9163085
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9138777, upper bound: 398.9157434
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9143552, upper bound: 398.9148838
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9135353, upper bound: 398.9168468
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9225973, upper bound: 398.9220129
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9225216, upper bound: 398.9275238
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9244839, upper bound: 398.9259259
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9265013, upper bound: 398.9259257
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773603, upper bound: 398.8792131
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773464, upper bound: 398.8785329
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8242168, upper bound: 398.8242297
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8242168, upper bound: 398.8242297
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9275594, upper bound: 398.9242903
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9231566, upper bound: 398.9284768
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9230373, upper bound: 398.9253653
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9259035
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9266710, upper bound: 398.9244848
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9239538, upper bound: 398.9272396
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9274437, upper bound: 398.9241853
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9273789, upper bound: 398.9263288
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9236712, upper bound: 398.9237799
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9236712, upper bound: 398.9237622
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9274139, upper bound: 398.9257963
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9245741, upper bound: 398.9243341
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9237958, upper bound: 398.9238287
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9237958, upper bound: 398.9238303
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9236180, upper bound: 398.9236180
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9236323, upper bound: 398.9236180
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9229512, upper bound: 398.9234051
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9231179, upper bound: 398.9229770
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9229345, upper bound: 398.9229207
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9236470, upper bound: 398.9227797
time: 0.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9153829, upper bound: 398.9163085
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9138777, upper bound: 398.9157434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9143552, upper bound: 398.9148838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9135353, upper bound: 398.9168468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9225973, upper bound: 398.9220129
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9225216, upper bound: 398.9275238
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9244839, upper bound: 398.9259259
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9265013, upper bound: 398.9259257
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.8773603, upper bound: 398.8792131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.8773464, upper bound: 398.8785329
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.8242168, upper bound: 398.8242297
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.8242168, upper bound: 398.8242297
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9275594, upper bound: 398.9242903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9231566, upper bound: 398.9284768
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9230373, upper bound: 398.9253653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9259035
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9266710, upper bound: 398.9244848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9239538, upper bound: 398.9272396
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9274437, upper bound: 398.9241853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9273789, upper bound: 398.9263288
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9236712, upper bound: 398.9237799
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9236712, upper bound: 398.9237622
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9274139, upper bound: 398.9257963
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9245741, upper bound: 398.9243341
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9237958, upper bound: 398.9238287
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9237958, upper bound: 398.9238303
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9236180, upper bound: 398.9236180
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9236323, upper bound: 398.9236180
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9229512, upper bound: 398.9234051
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9231179, upper bound: 398.9229770
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9229345, upper bound: 398.9229207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -398.9236470, upper bound: 398.9227797

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9104970, upper bound: 398.9098033
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9091209, upper bound: 398.9115179
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9094842, upper bound: 398.9125484
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9113939, upper bound: 398.9125484
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9143552, upper bound: 398.9131063
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9127113, upper bound: 398.9147812
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8801981, upper bound: 398.8803834
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8801032, upper bound: 398.8816609
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9219892, upper bound: 398.9219892
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9225973, upper bound: 398.9220129
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9219892, upper bound: 398.9265823
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9219892, upper bound: 398.9275222
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9241336, upper bound: 398.9259259
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9244839, upper bound: 398.9253564
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9259101, upper bound: 398.9223735
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9232829, upper bound: 398.9239674
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773600, upper bound: 398.8792131
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773603, upper bound: 398.8773464
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8772045, upper bound: 398.8783121
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8771471, upper bound: 398.8782342
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9228554, upper bound: 398.9230808
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9297249, upper bound: 398.9238787
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8270994, upper bound: 398.8236462
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8270994, upper bound: 398.8236462
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9215684, upper bound: 398.9219505
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9216802, upper bound: 398.9251723
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9220346
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9259035
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9061547, upper bound: 398.9059865
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9074300, upper bound: 398.9061264
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9060248, upper bound: 398.9069617
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9060103, upper bound: 398.9067358
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9273082, upper bound: 398.9241794
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9273082, upper bound: 398.9241283
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9218979, upper bound: 398.9245567
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9218806, upper bound: 398.9218943
time: 1.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9234716, upper bound: 398.9234750
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9234716, upper bound: 398.9234718
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9227399, upper bound: 398.9227999
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9227399, upper bound: 398.9227399
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9249158, upper bound: 398.9229176
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220106, upper bound: 398.9229202
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9233085, upper bound: 398.9233539
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9236728, upper bound: 398.9236632
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9234633, upper bound: 398.9234633
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9234633, upper bound: 398.9235319
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9228901, upper bound: 398.9228901
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9228901, upper bound: 398.9229197
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9209093, upper bound: 398.9209093
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9209093, upper bound: 398.9209093
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9228369, upper bound: 398.9228369
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9228369, upper bound: 398.9228369
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9211966, upper bound: 398.9222671
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9219063, upper bound: 398.9219146
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9226502, upper bound: 398.9227743
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9226502, upper bound: 398.9228075
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9227797, upper bound: 398.9227797
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9235178, upper bound: 398.9229207
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9229485, upper bound: 398.9220416
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9229422, upper bound: 398.9220416
time: 1.06 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9104970, upper bound: 398.9098033
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9091209, upper bound: 398.9115179
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9094842, upper bound: 398.9125484
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9113939, upper bound: 398.9125484
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9143552, upper bound: 398.9131063
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9127113, upper bound: 398.9147812
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8801981, upper bound: 398.8803834
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8801032, upper bound: 398.8816609
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9219892, upper bound: 398.9219892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9225973, upper bound: 398.9220129
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9219892, upper bound: 398.9265823
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9219892, upper bound: 398.9275222
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9241336, upper bound: 398.9259259
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9244839, upper bound: 398.9253564
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9259101, upper bound: 398.9223735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9232829, upper bound: 398.9239674
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8773600, upper bound: 398.8792131
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8773603, upper bound: 398.8773464
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8772045, upper bound: 398.8783121
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8771471, upper bound: 398.8782342
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9228554, upper bound: 398.9230808
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9297249, upper bound: 398.9238787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8270994, upper bound: 398.8236462
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.8270994, upper bound: 398.8236462
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9215684, upper bound: 398.9219505
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9216802, upper bound: 398.9251723
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9220346
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9259035
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9061547, upper bound: 398.9059865
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9074300, upper bound: 398.9061264
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9060248, upper bound: 398.9069617
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9060103, upper bound: 398.9067358
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9273082, upper bound: 398.9241794
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9273082, upper bound: 398.9241283
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9218979, upper bound: 398.9245567
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9218806, upper bound: 398.9218943
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9234716, upper bound: 398.9234750
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9234716, upper bound: 398.9234718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9227399, upper bound: 398.9227999
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9227399, upper bound: 398.9227399
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9249158, upper bound: 398.9229176
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9220106, upper bound: 398.9229202
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9233085, upper bound: 398.9233539
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9236728, upper bound: 398.9236632
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9234633, upper bound: 398.9234633
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9234633, upper bound: 398.9235319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9228901, upper bound: 398.9228901
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9228901, upper bound: 398.9229197
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9209093, upper bound: 398.9209093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9209093, upper bound: 398.9209093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9228369, upper bound: 398.9228369
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9228369, upper bound: 398.9228369
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9211966, upper bound: 398.9222671
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9219063, upper bound: 398.9219146
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9226502, upper bound: 398.9227743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9226502, upper bound: 398.9228075
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9227797, upper bound: 398.9227797
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9235178, upper bound: 398.9229207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9229485, upper bound: 398.9220416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -398.9229422, upper bound: 398.9220416

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8039361, upper bound: 398.8041139
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8039361, upper bound: 398.8041139
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870680, upper bound: 398.8885982
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870680, upper bound: 398.8885982
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9088135, upper bound: 398.9115869
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9088135, upper bound: 398.9090025
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9094842, upper bound: 398.9100376
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9099107, upper bound: 398.9094842
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9108673, upper bound: 398.9093219
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9093121, upper bound: 398.9097938
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9112658, upper bound: 398.9134086
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9112658, upper bound: 398.9112658
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792760, upper bound: 398.8794417
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792454, upper bound: 398.8792454
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792577, upper bound: 398.8806995
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8792577, upper bound: 398.8806995
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9213660, upper bound: 398.9213660
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9213660, upper bound: 398.9213660
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9099528, upper bound: 398.9099528
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9109229, upper bound: 398.9099528
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9204353, upper bound: 398.9250701
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9204353, upper bound: 398.9204353
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9208412, upper bound: 398.9260733
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9209900, upper bound: 398.9204353
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9237941, upper bound: 398.9259038
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9241181, upper bound: 398.9254716
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9233239, upper bound: 398.9249860
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9240465, upper bound: 398.9227797
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9250428, upper bound: 398.9223704
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9258915, upper bound: 398.9221762
time: 1.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9051886, upper bound: 398.9046140
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9043823, upper bound: 398.9043823
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773226, upper bound: 398.8790814
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773303, upper bound: 398.8789717
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8751076, upper bound: 398.8751076
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8751076, upper bound: 398.8751076
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8772045, upper bound: 398.8779125
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8771471, upper bound: 398.8783121
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8730824, upper bound: 398.8738508
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8729904, upper bound: 398.8734372
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833777, upper bound: 398.8843145
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833777, upper bound: 398.8843145
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9282841, upper bound: 398.9232204
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9290897, upper bound: 398.9218970
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8755643, upper bound: 398.8759392
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8755643, upper bound: 398.8760213
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9078254, upper bound: 398.9087121
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9097353, upper bound: 398.9091107
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9220346
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9220346
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9259035
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9222727
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9057130, upper bound: 398.9057130
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9058879, upper bound: 398.9057130
time: 1.13 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8039361, upper bound: 398.8041139
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8039361, upper bound: 398.8041139
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8870680, upper bound: 398.8885982
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8870680, upper bound: 398.8885982
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9088135, upper bound: 398.9115869
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9088135, upper bound: 398.9090025
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9094842, upper bound: 398.9100376
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9099107, upper bound: 398.9094842
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9108673, upper bound: 398.9093219
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9093121, upper bound: 398.9097938
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9112658, upper bound: 398.9134086
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9112658, upper bound: 398.9112658
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8792760, upper bound: 398.8794417
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8792454, upper bound: 398.8792454
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8792577, upper bound: 398.8806995
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8792577, upper bound: 398.8806995
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9213660, upper bound: 398.9213660
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9213660, upper bound: 398.9213660
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9099528, upper bound: 398.9099528
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9109229, upper bound: 398.9099528
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9204353, upper bound: 398.9250701
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9204353, upper bound: 398.9204353
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9208412, upper bound: 398.9260733
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9209900, upper bound: 398.9204353
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9237941, upper bound: 398.9259038
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9241181, upper bound: 398.9254716
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9233239, upper bound: 398.9249860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9240465, upper bound: 398.9227797
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9250428, upper bound: 398.9223704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9258915, upper bound: 398.9221762
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9051886, upper bound: 398.9046140
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9043823, upper bound: 398.9043823
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8773226, upper bound: 398.8790814
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8773303, upper bound: 398.8789717
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8751076, upper bound: 398.8751076
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8751076, upper bound: 398.8751076
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8772045, upper bound: 398.8779125
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8771471, upper bound: 398.8783121
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8730824, upper bound: 398.8738508
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8729904, upper bound: 398.8734372
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8833777, upper bound: 398.8843145
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8833777, upper bound: 398.8843145
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9282841, upper bound: 398.9232204
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9290897, upper bound: 398.9218970
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8755643, upper bound: 398.8759392
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.8755643, upper bound: 398.8760213
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9078254, upper bound: 398.9087121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9097353, upper bound: 398.9091107
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9220346
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9220346
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9259035
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9220346, upper bound: 398.9222727
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9057130, upper bound: 398.9057130
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.77
Output dim: 0, lower bound: -398.9058879, upper bound: 398.9057130
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9074300, upper bound: 398.9061264
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9060248, upper bound: 398.9069617
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9060103, upper bound: 398.9067358
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9273082, upper bound: 398.9241794
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9273082, upper bound: 398.9241283
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9218979, upper bound: 398.9245567
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9218806, upper bound: 398.9218943
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9234716, upper bound: 398.9234750
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9234716, upper bound: 398.9234718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9227399, upper bound: 398.9227999
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9227399, upper bound: 398.9227399
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9249158, upper bound: 398.9229176
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9220106, upper bound: 398.9229202
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9233085, upper bound: 398.9233539
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9236728, upper bound: 398.9236632
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9234633, upper bound: 398.9234633
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9234633, upper bound: 398.9235319
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9228901, upper bound: 398.9228901
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9228901, upper bound: 398.9229197
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9209093, upper bound: 398.9209093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9209093, upper bound: 398.9209093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9228369, upper bound: 398.9228369
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9228369, upper bound: 398.9228369
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9211966, upper bound: 398.9222671
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9219063, upper bound: 398.9219146
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9226502, upper bound: 398.9227743
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9226502, upper bound: 398.9228075
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9227797, upper bound: 398.9227797
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9235178, upper bound: 398.9229207
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9229485, upper bound: 398.9220416
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.77
Output dim: 0, lower bound: -398.9229422, upper bound: 398.9220416
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=482.57733154296875
rel_dist={0: [-398.9373570313671, 398.9373570313671]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8947380, upper bound: 398.8947380
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8947380, upper bound: 398.8947380
time: 0.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 0, lower bound: -398.8947380, upper bound: 398.8947380
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 0, lower bound: -398.8947380, upper bound: 398.8947380

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931628, upper bound: 398.8945526
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931628, upper bound: 398.8931628
time: 0.95 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930833, upper bound: 398.8932676
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8932676, upper bound: 398.8930980
time: 0.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 0, lower bound: -398.8931628, upper bound: 398.8945526
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 0, lower bound: -398.8931628, upper bound: 398.8931628
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 0, lower bound: -398.8930833, upper bound: 398.8932676
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.27
Output dim: 0, lower bound: -398.8932676, upper bound: 398.8930980

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931628, upper bound: 398.8945526
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930589, upper bound: 398.8933106
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8929600, upper bound: 398.8922952
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8924731, upper bound: 398.8922952
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919573, upper bound: 398.8932608
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919573, upper bound: 398.8929593
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8918428, upper bound: 398.8912409
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8912466, upper bound: 398.8916853
time: 0.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8931628, upper bound: 398.8945526
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8930589, upper bound: 398.8933106
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8929600, upper bound: 398.8922952
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8924731, upper bound: 398.8922952
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8919573, upper bound: 398.8932608
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8919573, upper bound: 398.8929593
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8918428, upper bound: 398.8912409
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -398.8912466, upper bound: 398.8916853

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930043, upper bound: 398.8944539
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930044, upper bound: 398.8938625
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8898254, upper bound: 398.8899723
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8898317, upper bound: 398.8899427
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8886860, upper bound: 398.8889564
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8894883, upper bound: 398.8889490
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8886860, upper bound: 398.8888409
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891479, upper bound: 398.8886841
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8900061, upper bound: 398.8920802
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8908079, upper bound: 398.8917043
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919981, upper bound: 398.8929593
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930599, upper bound: 398.8918316
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8898733, upper bound: 398.8894349
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891820, upper bound: 398.8894876
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905897, upper bound: 398.8908311
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903532, upper bound: 398.8909823
time: 1.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8930043, upper bound: 398.8944539
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8930044, upper bound: 398.8938625
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8898254, upper bound: 398.8899723
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8898317, upper bound: 398.8899427
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8886860, upper bound: 398.8889564
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8894883, upper bound: 398.8889490
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8886860, upper bound: 398.8888409
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8891479, upper bound: 398.8886841
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8900061, upper bound: 398.8920802
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8908079, upper bound: 398.8917043
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8919981, upper bound: 398.8929593
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8930599, upper bound: 398.8918316
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8898733, upper bound: 398.8894349
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8891820, upper bound: 398.8894876
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8905897, upper bound: 398.8908311
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.16
Output dim: 0, lower bound: -398.8903532, upper bound: 398.8909823

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8332331, upper bound: 398.8338447
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8329178, upper bound: 398.8338447
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8888292, upper bound: 398.8886649
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8886929, upper bound: 398.8888604
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8857359, upper bound: 398.8858623
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8857358, upper bound: 398.8857004
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8894114, upper bound: 398.8896833
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895762, upper bound: 398.8893602
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8890231, upper bound: 398.8886482
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8883867, upper bound: 398.8885910
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8887404, upper bound: 398.8887778
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8893139, upper bound: 398.8885157
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8886841, upper bound: 398.8888269
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891530, upper bound: 398.8888409
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8885157, upper bound: 398.8885157
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889709, upper bound: 398.8885157
time: 1.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8894783, upper bound: 398.8899318
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8894783, upper bound: 398.8905793
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8318542, upper bound: 398.8340147
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8318542, upper bound: 398.8340147
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847923, upper bound: 398.8848253
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8846532, upper bound: 398.8849305
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917922, upper bound: 398.8911609
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914681, upper bound: 398.8912707
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891799, upper bound: 398.8894349
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8896395, upper bound: 398.8891888
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889610, upper bound: 398.8892271
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889610, upper bound: 398.8892092
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8879291, upper bound: 398.8882465
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8882063, upper bound: 398.8885493
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903532, upper bound: 398.8909823
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903201, upper bound: 398.8908174
time: 1.25 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8332331, upper bound: 398.8338447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8329178, upper bound: 398.8338447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8888292, upper bound: 398.8886649
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8886929, upper bound: 398.8888604
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8857359, upper bound: 398.8858623
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8857358, upper bound: 398.8857004
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8894114, upper bound: 398.8896833
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8895762, upper bound: 398.8893602
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8890231, upper bound: 398.8886482
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8883867, upper bound: 398.8885910
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8887404, upper bound: 398.8887778
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8893139, upper bound: 398.8885157
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8886841, upper bound: 398.8888269
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8891530, upper bound: 398.8888409
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8885157, upper bound: 398.8885157
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8889709, upper bound: 398.8885157
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8894783, upper bound: 398.8899318
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8894783, upper bound: 398.8905793
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8318542, upper bound: 398.8340147
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8318542, upper bound: 398.8340147
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8847923, upper bound: 398.8848253
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8846532, upper bound: 398.8849305
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8917922, upper bound: 398.8911609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8914681, upper bound: 398.8912707
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8891799, upper bound: 398.8894349
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8896395, upper bound: 398.8891888
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8889610, upper bound: 398.8892271
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8889610, upper bound: 398.8892092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8879291, upper bound: 398.8882465
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8882063, upper bound: 398.8885493
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8903532, upper bound: 398.8909823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -398.8903201, upper bound: 398.8908174

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8883854, upper bound: 398.8883880
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8885534, upper bound: 398.8883854
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838781, upper bound: 398.8838367
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838303, upper bound: 398.8839120
time: 1.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8840886, upper bound: 398.8842818
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8840100, upper bound: 398.8842604
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8843579, upper bound: 398.8846117
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848721, upper bound: 398.8844008
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8890092, upper bound: 398.8892458
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8890092, upper bound: 398.8890092
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8863324, upper bound: 398.8863166
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8863166, upper bound: 398.8863166
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8873036, upper bound: 398.8871007
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8869492, upper bound: 398.8871004
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8871509, upper bound: 398.8873344
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8871509, upper bound: 398.8872923
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8823555, upper bound: 398.8823158
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8822941, upper bound: 398.8823433
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8879960, upper bound: 398.8873070
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8873658, upper bound: 398.8873070
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889171, upper bound: 398.8888060
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889552, upper bound: 398.8888057
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8791820, upper bound: 398.8794804
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8791820, upper bound: 398.8794804
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794988, upper bound: 398.8788408
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794988, upper bound: 398.8788408
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889533, upper bound: 398.8885045
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8889492, upper bound: 398.8885045
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8873161, upper bound: 398.8877142
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8873161, upper bound: 398.8878130
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8888577, upper bound: 398.8900264
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8888577, upper bound: 398.8888577
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845913, upper bound: 398.8846601
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845845, upper bound: 398.8846506
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8726578, upper bound: 398.8731973
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8726578, upper bound: 398.8731973
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8904800, upper bound: 398.8899938
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8907262, upper bound: 398.8897183
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8050567, upper bound: 398.8050567
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8050567, upper bound: 398.8050567
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8884432, upper bound: 398.8883651
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8887998, upper bound: 398.8883557
time: 1.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8869376, upper bound: 398.8869376
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8872145, upper bound: 398.8869423
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315
1: -197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482
2: -197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371
3: -234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619
4: -201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876943, upper bound: 398.8879769
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876943, upper bound: 398.8879989
time: 1.22 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8883854, upper bound: 398.8883880
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8885534, upper bound: 398.8883854
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8838781, upper bound: 398.8838367
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8838303, upper bound: 398.8839120
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8840886, upper bound: 398.8842818
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8840100, upper bound: 398.8842604
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8843579, upper bound: 398.8846117
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8848721, upper bound: 398.8844008
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8890092, upper bound: 398.8892458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8890092, upper bound: 398.8890092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8863324, upper bound: 398.8863166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8863166, upper bound: 398.8863166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8873036, upper bound: 398.8871007
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8869492, upper bound: 398.8871004
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8871509, upper bound: 398.8873344
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8871509, upper bound: 398.8872923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8823555, upper bound: 398.8823158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8822941, upper bound: 398.8823433
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8879960, upper bound: 398.8873070
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8873658, upper bound: 398.8873070
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8889171, upper bound: 398.8888060
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8889552, upper bound: 398.8888057
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8791820, upper bound: 398.8794804
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8791820, upper bound: 398.8794804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8794988, upper bound: 398.8788408
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8794988, upper bound: 398.8788408
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8889533, upper bound: 398.8885045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8889492, upper bound: 398.8885045
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8873161, upper bound: 398.8877142
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8873161, upper bound: 398.8878130
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8888577, upper bound: 398.8900264
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8888577, upper bound: 398.8888577
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8845913, upper bound: 398.8846601
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8845845, upper bound: 398.8846506
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8726578, upper bound: 398.8731973
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8726578, upper bound: 398.8731973
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8904800, upper bound: 398.8899938
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8907262, upper bound: 398.8897183
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8050567, upper bound: 398.8050567
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8050567, upper bound: 398.8050567
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8884432, upper bound: 398.8883651
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8887998, upper bound: 398.8883557
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8869376, upper bound: 398.8869376
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8872145, upper bound: 398.8869423
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8876943, upper bound: 398.8879769
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -398.8876943, upper bound: 398.8879989
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.39
Output dim: 0, lower bound: -398.8889610, upper bound: 398.8892092
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.39
Output dim: 0, lower bound: -398.8879291, upper bound: 398.8882465
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.39
Output dim: 0, lower bound: -398.8882063, upper bound: 398.8885493
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.39
Output dim: 0, lower bound: -398.8903532, upper bound: 398.8909823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.39
Output dim: 0, lower bound: -398.8903201, upper bound: 398.8908174
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=482.57733154296875
rel_dist={0: [-398.93352538884415, 398.93352538884415]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1097.10 seconds
